"""W124 Lane-α environment feasibility probe (NOT a benchmark; diagnostics only).

Decides, on THIS host, whether a real transformer-native code-substrate path is
runnable or whether Lane α must fall back to the deterministic stub/contract path.
Prints a machine-greppable verdict block. No NIM. No network. No claims.
"""
from __future__ import annotations

import inspect
import json
import sys
import traceback

V = {"py": sys.version.split()[0]}


def _try(label, fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - diagnostic probe
        V[label + "_err"] = f"{type(e).__name__}: {e}"
        return None


# --- torch / transformers ---
def _tf():
    import torch
    import transformers

    V["torch"] = torch.__version__
    V["transformers"] = transformers.__version__
    V["cuda"] = bool(torch.cuda.is_available())
    V["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return True


_try("tf", _tf)

# --- AST/stub substrate (no model) ---
SRC = (
    "import sys\n\n"
    "def solve():\n"
    "    n = int(input())\n"
    "    total = 0\n"
    "    for _ in range(n):\n"
    "        total += int(input())\n"
    "    print(total)\n\n"
    "def main():\n"
    "    solve()\n\n"
    "main()\n"
)


def _ast():
    from coordpy.code_substrate_v1 import (
        extract_function_boundaries_v1,
        encode_source_with_stub_v1,
        W87_CODE_STUB_EMBEDDING_DIM,
    )

    V["ast_import"] = "ok"
    V["stub_dim"] = W87_CODE_STUB_EMBEDDING_DIM
    sig = str(inspect.signature(extract_function_boundaries_v1))
    V["extract_sig"] = sig
    kw = "source" in sig
    bounds = extract_function_boundaries_v1(source=SRC) if kw else extract_function_boundaries_v1(SRC)
    V["n_boundaries"] = len(bounds)
    V["boundary_names"] = [getattr(b, "name", None) or b.to_dict().get("name") for b in bounds][:6]
    esig = str(inspect.signature(encode_source_with_stub_v1))
    V["encode_stub_sig"] = esig
    return True


_try("ast", _ast)

# --- real transformer encoder (distilgpt2) via transformers_runtime_v1 ---
def _runtime():
    from coordpy import transformers_runtime_v1 as TR

    V["runtime_default_model"] = getattr(TR, "W80_TRANSFORMERS_DEFAULT_MODEL_NAME", None)
    # discover the constructor + its parameters
    cls = TR.TransformersRuntimeV1
    V["runtime_ctor_sig"] = str(inspect.signature(cls.__init__))
    # discover any builder/factory
    builders = [n for n in dir(TR) if "build" in n.lower() or "probe" in n.lower() or "load" in n.lower()]
    V["runtime_builders"] = builders[:12]
    return True


_try("runtime", _runtime)


def _load_distilgpt2():
    # Direct, minimal real forward pass to prove the host can read hidden states
    # from a REAL pretrained transformer at all (independent of repo wiring).
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, output_hidden_states=True)
    model.eval()
    ids = tok(SRC, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids)
    hs = out.hidden_states  # tuple(n_layers+1) of [1, seq, hidden]
    V["distilgpt2_n_hidden_layers"] = len(hs)
    V["distilgpt2_hidden_dim"] = int(hs[-1].shape[-1])
    V["distilgpt2_seq_len"] = int(hs[-1].shape[1])
    V["distilgpt2_last_layer_mean"] = round(float(hs[-1].mean()), 6)
    return True


_try("distilgpt2", _load_distilgpt2)

print("=== W124_FEASIBILITY_VERDICT ===")
print(json.dumps(V, indent=2))
print("=== END ===")
