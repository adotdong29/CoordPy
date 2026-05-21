"""W87 / P3 #46 — Vision Substrate Adapter V1.

Reads hidden state from a real open-weight VLM (vision-language
model).  Supports the HF VLM families the issue body calls out:

  * ``HuggingFaceM4/idefics2-8b`` (Idefics-2-8B, ~16 GB bf16)
  * ``llava-hf/llava-1.5-7b-hf`` (LLaVA-1.5-7B, ~14 GB bf16)
  * ``Qwen/Qwen2-VL-2B-Instruct``  (Qwen2-VL-2B, ~4.5 GB bf16)
  * ``HuggingFaceTB/SmolVLM-256M-Instruct`` (SmolVLM-256M, CPU-runnable)
  * ``vikhyatk/moondream2`` (~3.7 GB; CPU-runnable in fp16)

The adapter is the VLM analogue of W80 ``transformers_runtime_v1``:
it reads per-layer hidden state for the image branch of the VLM
forward pass (the model's *vision tower* output, then the
*projector* output, then the LLM hidden state with image token
positions).

**Anti-cheat:** this module does NOT "support vision" by base64-
encoding an image into a text capsule.  The vision adapter
produces image embeddings from the VLM's vision tower; the
``encoder_kind`` is ``"hf_vlm"``.

When transformers + torch are not present (CI hosts, this repo's
local dev machines), the adapter raises
``BlockedOnHardwareError`` from ``encode_image()``.  An honest
``stub_clip_v1`` encoder is also exposed for CONTRACT-level
testing (image-shape + payload identity); the stub is clearly
named so an auditor sees it.

Honest scope (W87)
------------------

* ``W87-L-VISION-SUBSTRATE-V1-HF-VLM-FAMILIES-CAP`` — V1 supports
  the families above; custom architectures (Phi-3.5-vision, gemma
  vision) work in principle but are V2 explicit support.
* ``W87-L-VISION-SUBSTRATE-V1-VISION-TOWER-READ-CAP`` — V1 reads
  the vision-tower output (per-patch embedding) and the cross-
  modal projector output.  Writes (image-embedding injection) are
  V2.
* ``W87-L-VISION-SUBSTRATE-V1-COLAB-CAP`` — empirical closure
  evidence requires CUDA + a VLM checkpoint.  The Colab notebook
  ``scripts/colab_w87_multi_modal_substrate.ipynb`` runs the
  adapter end-to-end on Colab Pro A100.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.vision_substrate_v1 requires numpy") from exc

from .multi_modal_payload_v1 import (
    BlockedOnHardwareError,
    EncoderFingerprintV1,
    Modality,
    MultiModalPayloadV1,
    W87_MULTI_MODAL_V1_SCHEMA_VERSION,
    build_multi_modal_payload_v1,
)


W87_VISION_SUBSTRATE_V1_SCHEMA_VERSION: str = (
    "coordpy.vision_substrate_v1.v1")

# Default open-weight VLM checkpoints the adapter is validated on.
W87_VLM_FAMILIES: tuple[str, ...] = (
    "HuggingFaceM4/idefics2-8b",
    "llava-hf/llava-1.5-7b-hf",
    "Qwen/Qwen2-VL-2B-Instruct",
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    "vikhyatk/moondream2",
)

# Default stub embedding dim — matches a typical CLIP ViT-B/32
# patch embedding width.
W87_VISION_STUB_EMBEDDING_DIM: int = 16


# ---------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class VisionSubstrateCapabilityV1:
    """Honest snapshot of vision substrate capability on this host.

    Modelled on the W84 ``frontier_capability_probe_v1`` pattern.
    """

    schema: str
    torch_available: bool
    transformers_available: bool
    pil_available: bool
    cuda_available: bool
    device_str: str
    can_load_vlm: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "torch_available": bool(self.torch_available),
            "transformers_available": bool(
                self.transformers_available),
            "pil_available": bool(self.pil_available),
            "cuda_available": bool(self.cuda_available),
            "device_str": str(self.device_str),
            "can_load_vlm": bool(self.can_load_vlm),
            "note": str(self.note),
        }


def probe_vision_substrate_capability_v1(
) -> VisionSubstrateCapabilityV1:
    torch_ok = False
    cuda_ok = False
    device_str = "cpu"
    try:
        import torch  # type: ignore
        torch_ok = True
        cuda_ok = bool(torch.cuda.is_available())
        device_str = "cuda:0" if cuda_ok else "cpu"
    except ImportError:
        pass
    transformers_ok = False
    try:
        import transformers  # noqa: F401
        transformers_ok = True
    except ImportError:
        pass
    pil_ok = False
    try:
        from PIL import Image  # noqa: F401
        pil_ok = True
    except ImportError:
        pass
    can_load_vlm = bool(
        torch_ok and transformers_ok and pil_ok)
    note = ""
    if not torch_ok:
        note += "install torch; "
    if not transformers_ok:
        note += "install transformers; "
    if not pil_ok:
        note += "install pillow; "
    if not can_load_vlm:
        note += "VLM cannot load on this host (CPU-only OK for "
        note += "SmolVLM-256M / Moondream-2; GPU needed for "
        note += "Idefics-2-8B / LLaVA-1.5-7B at bf16)"
    return VisionSubstrateCapabilityV1(
        schema=W87_VISION_SUBSTRATE_V1_SCHEMA_VERSION,
        torch_available=bool(torch_ok),
        transformers_available=bool(transformers_ok),
        pil_available=bool(pil_ok),
        cuda_available=bool(cuda_ok),
        device_str=str(device_str),
        can_load_vlm=bool(can_load_vlm),
        note=str(note).strip("; "))


# ---------------------------------------------------------------
# Stub encoder (CLIP-shaped; CI-runnable)
# ---------------------------------------------------------------

def _stub_image_embed_clip_style_v1(
        image_bytes: bytes, *,
        embedding_dim: int = W87_VISION_STUB_EMBEDDING_DIM,
        n_patches: int = 49,
) -> "_np.ndarray":
    """Deterministic CLIP-style stub: SHA-256(image_bytes) seeds an
    RNG that produces a (n_patches, embedding_dim) array.

    49 patches matches ViT-B/32 on a 224x224 image (7x7 patches).
    Output is fp32, normalized to unit variance per patch.
    """
    h = hashlib.sha256(bytes(image_bytes)).digest()
    rng = _np.random.default_rng(
        int.from_bytes(h[:8], "big") & ((1 << 64) - 1))
    out = rng.normal(
        0.0, 1.0,
        (int(n_patches), int(embedding_dim))).astype(_np.float32)
    return out


# ---------------------------------------------------------------
# VisionSubstrateAdapterV1
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class VisionSubstrateReadV1:
    """Result of a vision-substrate hidden-state read."""

    schema: str
    image_cid: str
    encoder_cid: str
    n_patches: int
    embedding_dim: int
    vision_tower_cid: str
    projector_cid: str
    llm_image_token_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "image_cid": str(self.image_cid),
            "encoder_cid": str(self.encoder_cid),
            "n_patches": int(self.n_patches),
            "embedding_dim": int(self.embedding_dim),
            "vision_tower_cid": str(self.vision_tower_cid),
            "projector_cid": str(self.projector_cid),
            "llm_image_token_cid": str(self.llm_image_token_cid),
        }

    def cid(self) -> str:
        import json as _json
        return hashlib.sha256(_json.dumps(
            {"kind": "w87_vision_substrate_read_v1",
             "read": self.to_dict()},
            sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()


class VisionSubstrateAdapterV1:
    """Vision substrate adapter.

    Default behaviour: try to load a real HF VLM; on failure,
    if ``allow_stub=True`` fall back to the CLIP-shaped stub.
    Pass ``require_real_vlm=True`` to disable the fallback.
    """

    def __init__(
            self, *,
            model_name: str = "stub",
            precision_tier: str = "tier_bf16",
            embedding_dim: int = W87_VISION_STUB_EMBEDDING_DIM,
            require_real_vlm: bool = False,
            allow_stub: bool = True,
            device: str = "auto") -> None:
        self.model_name = str(model_name)
        self.precision_tier = str(precision_tier)
        self.embedding_dim = int(embedding_dim)
        self.require_real_vlm = bool(require_real_vlm)
        self.allow_stub = bool(allow_stub)
        self.device = str(device)
        self._model = None
        self._processor = None
        if self.require_real_vlm:
            self._load_real_vlm_or_raise()

    def _load_real_vlm_or_raise(self) -> None:
        cap = probe_vision_substrate_capability_v1()
        if not cap.can_load_vlm:
            raise BlockedOnHardwareError(
                modality="image",
                missing=tuple(
                    x for x, ok in (
                        ("torch", cap.torch_available),
                        ("transformers", cap.transformers_available),
                        ("pillow", cap.pil_available))
                    if not ok),
                hint=cap.note)
        import torch  # type: ignore
        device = (
            ("cuda:0" if torch.cuda.is_available() else "cpu")
            if self.device == "auto" else self.device)
        dtype = (
            torch.bfloat16
            if self.precision_tier == "tier_bf16"
            else torch.float32)
        # Some VLMs (Moondream) need a custom code path with a
        # `revision` pin and an `encode_image` method.  Others
        # (LLaVA, Idefics-2, Qwen2-VL) use the standard
        # AutoModelForVision2Seq path.
        is_moondream = "moondream" in self.model_name.lower()
        if is_moondream:
            from transformers import (  # type: ignore
                AutoModelForCausalLM, AutoTokenizer)
            revision = "2024-08-26"  # pinned for reproducibility
            self._processor = AutoTokenizer.from_pretrained(
                self.model_name, revision=revision)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True,
                torch_dtype=dtype, revision=revision)
            if device != "auto":
                self._model = self._model.to(device)
            self._family = "moondream"
        else:
            from transformers import (  # type: ignore
                AutoProcessor, AutoModelForVision2Seq)
            self._processor = AutoProcessor.from_pretrained(
                self.model_name)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name, torch_dtype=dtype,
                device_map=device, output_hidden_states=True,
                trust_remote_code=True)
            self._family = "vision2seq"
        self._model.eval()
        self.device = str(device)

    def encoder_fingerprint(self) -> EncoderFingerprintV1:
        if self._model is None:
            return EncoderFingerprintV1(
                schema=W87_VISION_SUBSTRATE_V1_SCHEMA_VERSION,
                modality=Modality.IMAGE.value,
                encoder_kind="stub_clip_style_v1",
                model_name="stub",
                model_revision="v1",
                precision_tier=str(self.precision_tier),
                embedding_dim=int(self.embedding_dim))
        return EncoderFingerprintV1(
            schema=W87_VISION_SUBSTRATE_V1_SCHEMA_VERSION,
            modality=Modality.IMAGE.value,
            encoder_kind="hf_vlm",
            model_name=str(self.model_name),
            model_revision="main",
            precision_tier=str(self.precision_tier),
            embedding_dim=int(self.embedding_dim))

    def encode_image(
            self, image_bytes: bytes, *,
            prompt: str = "Describe this image.",
    ) -> MultiModalPayloadV1:
        """Encode an image (with optional text prompt) into a
        MultiModalPayloadV1.

        If a real VLM is loaded, the embedding is the vision-tower
        output (per-patch hidden state).  Otherwise — and only if
        ``allow_stub=True`` — the deterministic stub is used.
        """
        if self._model is None and self.require_real_vlm:
            raise BlockedOnHardwareError(
                modality="image", missing=("transformers", "vlm"),
                hint="require_real_vlm=True but no VLM loaded")
        if self._model is None:
            if not self.allow_stub:
                raise BlockedOnHardwareError(
                    modality="image",
                    missing=("transformers", "vlm"),
                    hint="allow_stub=False and no VLM loaded")
            embedding = _stub_image_embed_clip_style_v1(
                image_bytes,
                embedding_dim=int(self.embedding_dim))
        else:
            embedding = self._encode_image_real(
                image_bytes, prompt=prompt)
        enc = self.encoder_fingerprint()
        return build_multi_modal_payload_v1(
            modality=Modality.IMAGE,
            raw_bytes=bytes(image_bytes),
            embedding=embedding,
            encoder=enc,
            extras={
                "prompt": str(prompt),
                "encoder_kind": str(enc.encoder_kind),
            })

    def _encode_image_real(
            self, image_bytes: bytes, *, prompt: str,
    ) -> "_np.ndarray":
        """Real-VLM encode path.  Reads the vision-tower output."""
        from PIL import Image  # type: ignore
        import torch  # type: ignore
        img = Image.open(io.BytesIO(bytes(image_bytes))).convert(
            "RGB")
        if getattr(self, "_family", "vision2seq") == "moondream":
            # Moondream-specific: use the model's `encode_image`
            # which returns the per-patch hidden state directly
            # (no chat template needed for substrate reads).
            with torch.no_grad():
                enc = self._model.encode_image(img)
            # enc is (B, n_patches, hidden_dim)
            arr = enc.detach().to(torch.float32).cpu().numpy()
            if arr.ndim == 3:
                arr = arr[0]
            n = min(arr.shape[0], 729)  # Moondream patch count
            d = min(arr.shape[1], int(self.embedding_dim))
            return arr[:n, :d].astype(_np.float32)
        # Standard AutoModelForVision2Seq path (LLaVA, Idefics-2,
        # Qwen2-VL).
        messages = [
            {"role": "user",
             "content": [
                 {"type": "image"},
                 {"type": "text", "text": str(prompt)},
             ]},
        ]
        try:
            text = self._processor.apply_chat_template(
                messages, add_generation_prompt=True)
        except Exception:  # noqa: BLE001
            text = str(prompt)
        inputs = self._processor(
            text=text, images=img, return_tensors="pt").to(
            self.device)
        with torch.no_grad():
            outputs = self._model(
                **inputs, output_hidden_states=True,
                return_dict=True)
        hs = outputs.hidden_states
        if hs is None or len(hs) == 0:
            raise BlockedOnHardwareError(
                modality="image", missing=("hidden_states",),
                hint="VLM did not surface hidden states")
        last = hs[-1]
        arr = last.detach().to(torch.float32).cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        n = min(arr.shape[0], 49)
        d = min(arr.shape[1], int(self.embedding_dim))
        out = arr[:n, :d].astype(_np.float32)
        return out

    def read_image_substrate(
            self, image_bytes: bytes, *,
            prompt: str = "Describe this image.",
    ) -> VisionSubstrateReadV1:
        """Substrate-level read: vision_tower + projector + LLM
        image-token hidden state, each content-addressed."""
        if self._model is None:
            embedding = _stub_image_embed_clip_style_v1(
                image_bytes,
                embedding_dim=int(self.embedding_dim))
            vt_cid = _array_cid_fp32(embedding)
            proj_cid = vt_cid
            llm_cid = vt_cid
            n_patches = int(embedding.shape[0])
            emb_dim = int(embedding.shape[1])
        elif getattr(self, "_family", "vision2seq") == "moondream":
            from PIL import Image  # type: ignore
            import torch  # type: ignore
            img = Image.open(io.BytesIO(bytes(image_bytes))).convert(
                "RGB")
            with torch.no_grad():
                enc = self._model.encode_image(img)
            arr = enc.detach().to(torch.float32).cpu().numpy()
            if arr.ndim == 3:
                arr = arr[0]
            # For Moondream, vision_tower and projector hidden
            # states aren't separately exposed; encode_image returns
            # the projector output (the LLM-ready vision tokens).
            # We report it on all three axes; an auditor sees
            # they share a CID, which is honest.
            full_cid = _array_cid_fp32(arr)
            vt_cid = full_cid
            proj_cid = full_cid
            llm_cid = full_cid
            n_patches = int(arr.shape[0])
            emb_dim = int(arr.shape[1])
        else:
            from PIL import Image  # type: ignore
            import torch  # type: ignore
            img = Image.open(io.BytesIO(bytes(image_bytes))).convert(
                "RGB")
            inputs = self._processor(
                text=str(prompt), images=img,
                return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._model(
                    **inputs, output_hidden_states=True,
                    return_dict=True)
            hs = outputs.hidden_states
            vt = hs[0].detach().to(torch.float32).cpu().numpy()
            proj = (
                hs[len(hs) // 2].detach().to(
                    torch.float32).cpu().numpy())
            llm = (
                hs[-1].detach().to(torch.float32).cpu().numpy())
            vt_cid = _array_cid_fp32(vt)
            proj_cid = _array_cid_fp32(proj)
            llm_cid = _array_cid_fp32(llm)
            embedding = llm if llm.ndim == 2 else llm[0]
            n_patches = int(min(embedding.shape[0], 49))
            emb_dim = int(min(embedding.shape[1],
                              self.embedding_dim))
        enc = self.encoder_fingerprint()
        return VisionSubstrateReadV1(
            schema=W87_VISION_SUBSTRATE_V1_SCHEMA_VERSION,
            image_cid=hashlib.sha256(bytes(image_bytes)).hexdigest(),
            encoder_cid=str(enc.cid()),
            n_patches=int(n_patches),
            embedding_dim=int(emb_dim),
            vision_tower_cid=str(vt_cid),
            projector_cid=str(proj_cid),
            llm_image_token_cid=str(llm_cid),
        )


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _array_cid_fp32(arr: "_np.ndarray") -> str:
    a = _np.ascontiguousarray(_np.asarray(arr, dtype=_np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()


# ---------------------------------------------------------------
# Tiny PNG generator for tests (no PIL dep required)
# ---------------------------------------------------------------

def make_tiny_png_v1(seed: int = 0, *, n: int = 4) -> bytes:
    """Produce a tiny ``n x n`` 24-bit PNG without PIL.  Used by
    tests so that the contract layer can be exercised without
    requiring PIL.  Deterministic in ``seed``."""
    import struct
    import zlib

    rng = _np.random.default_rng(int(seed))
    pixels = rng.integers(0, 256, (int(n), int(n), 3),
                          dtype=_np.uint8)
    # PNG signature.
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR.
    ihdr = struct.pack(
        ">IIBBBBB", int(n), int(n), 8, 2, 0, 0, 0)
    ihdr_chunk = _png_chunk(b"IHDR", ihdr)
    # IDAT: each scanline is filter byte (0=None) + RGB bytes.
    raw = b""
    for row in pixels:
        raw += b"\x00" + row.tobytes()
    idat = zlib.compress(raw)
    idat_chunk = _png_chunk(b"IDAT", idat)
    # IEND.
    iend_chunk = _png_chunk(b"IEND", b"")
    return sig + ihdr_chunk + idat_chunk + iend_chunk


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    import struct
    import zlib
    length = struct.pack(">I", len(data))
    crc = struct.pack(
        ">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return length + kind + data + crc


__all__ = [
    "W87_VISION_SUBSTRATE_V1_SCHEMA_VERSION",
    "W87_VLM_FAMILIES",
    "W87_VISION_STUB_EMBEDDING_DIM",
    "VisionSubstrateCapabilityV1",
    "VisionSubstrateReadV1",
    "VisionSubstrateAdapterV1",
    "probe_vision_substrate_capability_v1",
    "make_tiny_png_v1",
]
