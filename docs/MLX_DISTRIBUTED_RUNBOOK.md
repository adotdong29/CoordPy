# MLX Distributed Inference — Two-Mac Operator Runbook

> Operator-facing bring-up runbook for a single sharded model
> spanning two Apple Silicon Macs via MLX distributed inference,
> targeted at use as a CoordPy LLM backend
> (`MLXDistributedBackend`). Last touched: 2026-04-26 (SDK v3.6).
>
> CoordPy **does not** auto-bring-up the cluster; this runbook is
> the documented out-of-band procedure. The CoordPy-side
> integration boundary is one HTTP-client class — see
> `vision_mvp/coordpy/llm_backend.py` and `docs/archive/coordpy-milestones/RESULTS_COORDPY_DISTRIBUTED.md`.

## 0. Preconditions

* Two Apple Silicon Macs on the same LAN. The setup this repo
  targets is two M3 Pro 36 GB Macs at, e.g.,
  `192.168.12.191` (head, rank 0) and `192.168.12.248`
  (worker, rank 1).
* SSH access between hosts (the existing
  `cluster_secure/ssh_config` from
  `fin-ground-test/SECURE_CLUSTER_SETUP.md` is reused).
* Homebrew available on both Macs.
* Python 3.10+ on both Macs.
* Roughly 50 GB free disk on each Mac (model weights cache).

## 1. One-time per-host install

On **each** Mac:

```bash
# OpenMPI for the launcher.
brew install open-mpi

# Apple's MLX + the LM helper layer.
python3 -m venv ~/.venvs/mlx
source ~/.venvs/mlx/bin/activate
pip install --upgrade pip
pip install "mlx>=0.21" "mlx-lm>=0.20"

# Sanity check.
python3 -c "import mlx.core as mx; print(mx.metal.is_available())"
# expect: True
```

Verify MPI works between the two Macs (replace usernames /
hosts with yours):

```bash
mpirun --hostfile ./hosts \
  -np 2 \
  --mca btl_tcp_if_include 192.168.12.0/24 \
  hostname
# expect: two lines, one per Mac
```

`./hosts` content (one line per host, format follows OpenMPI):

```
mac1.local slots=1
mac2.local slots=1
```

## 2. One-time per-model conversion

Pick a model. The honest target on 2×36 GB is a 70B-class model
in 4-bit:

```bash
# On each Mac (model cache must exist on every rank).
python3 -m mlx_lm.convert \
  --hf-path meta-llama/Llama-3.3-70B-Instruct \
  --mlx-path ~/.cache/mlx/Llama-3.3-70B-Instruct-4bit \
  -q --q-bits 4
```

Smaller drop-ins:

* `mlx-community/Llama-3.3-70B-Instruct-4bit` — pre-quantised,
  pull from HF directly:

  ```bash
  pip install huggingface_hub
  huggingface-cli download \
    mlx-community/Llama-3.3-70B-Instruct-4bit \
    --local-dir ~/.cache/mlx/Llama-3.3-70B-Instruct-4bit
  ```

* `mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` (smaller,
  also fits on one 36 GB Mac in 4-bit, but sharding gives KV
  / context headroom).

## 3. Bring up the sharded server

From the **head** Mac (rank 0):

```bash
source ~/.venvs/mlx/bin/activate

mpirun --hostfile ./hosts \
  -np 2 \
  --mca btl_tcp_if_include 192.168.12.0/24 \
  -x PATH \
  python3 -m mlx_lm.server \
    --model ~/.cache/mlx/Llama-3.3-70B-Instruct-4bit \
    --host 0.0.0.0 \
    --port 8080
```

`mlx_lm.server` will:

1. Initialise `mx.distributed` over MPI; rank 0 and rank 1
   negotiate the tensor / pipeline split.
2. Bind an OpenAI-compatible HTTP server to the head's
   `0.0.0.0:8080` (the worker rank does not bind a server —
   only the head does).
3. Serve `/v1/chat/completions` requests; each request triggers
   collective ops between the two ranks for the forward pass.

**Smoke check** (from the controller, NOT the head):

```bash
curl -s http://192.168.12.191:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "Llama-3.3-70B-Instruct-4bit",
        "messages": [{"role":"user","content":"reply with OK"}],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": false
      }'
# expect: {"choices":[{"message":{"content":"OK"...
```

## 4. Point CoordPy at it

```python
from vision_mvp.coordpy import (
    SweepSpec, run_sweep, MLXDistributedBackend,
    CapsuleNativeRunContext,
)

backend = MLXDistributedBackend(
    model="Llama-3.3-70B-Instruct-4bit",
    base_url="http://192.168.12.191:8080",
    timeout=600.0,
)

spec = SweepSpec(
    mode="real",
    jsonl="vision_mvp/tasks/data/swe_lite_style_bank.jsonl",
    sandbox="in_process",
    parser_modes=("strict", "robust"),
    apply_modes=("strict",),
    n_distractors=(0,),
    n_instances=10,
    model="Llama-3.3-70B-Instruct-4bit",
    endpoint="http://192.168.12.191:8080",
    acknowledge_heavy=True,
)

ctx = CapsuleNativeRunContext()
ctx.start_run(profile_name="mlx_distributed_70b", profile_dict={})
block = run_sweep(spec, ctx=ctx, llm_backend=backend)
print("backend:", block["backend"])           # MLXDistributedBackend
print("cells:",   len(block["cells"]))
print("model:",   block["model"])
```

Capsule chain seals end-to-end: PROMPT / LLM_RESPONSE /
PARSE_OUTCOME / PATCH_PROPOSAL / TEST_VERDICT, lifecycle audit
L-1..L-11 holds, root_cid is reproducible across runs (with
`RunSpec(deterministic=True)` modulo the LLM's own determinism).

## 5. Run the real cross-model parser-boundary study against the
70B class

```bash
python3 -m vision_mvp.experiments.parser_boundary_real_llm \
  --endpoint http://192.168.12.191:8080 \
  --n-instances 10 \
  --out /tmp/coordpy-distributed/real_cross_model_70b.json
```

(Modify the experiment's `DEFAULT_MODELS` list to include the
70B MLX tag plus the existing Ollama 14B baseline; the harness
issues the requests against whichever HTTP server matches the
configured endpoint per model.)

## 6. Tear-down

```bash
# On the head Mac:
pkill -f mlx_lm.server
# Worker rank exits when MPI sees rank 0 disappear.
```

## 7. Common failure modes

| symptom                                         | cause                                                                | fix                                                                    |
| ----------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `mpirun: orted launch failed`                   | SSH key / hostfile mismatch                                          | run `ssh mac2.local hostname` first; verify `slots=1` per line          |
| Worker rank OOMs                                | shard size > worker memory                                           | bump `--q-bits` to 4; pick a smaller model                             |
| `mx.distributed.init` hangs                     | TCP interface mismatch                                               | set `--mca btl_tcp_if_include 192.168.12.0/24` to your subnet         |
| HTTP 500 from `/v1/chat/completions`            | model loaded only on rank 0                                          | each rank must have the converted MLX model on disk locally           |
| CoordPy hangs at first call                       | cold model load > CoordPy `timeout`                                    | bump `MLXDistributedBackend(timeout=900.0)` for first call             |

## 8. What this runbook does NOT do

* Bring up the existing `aspen_cluster` Ollama harness — see
  `fin-ground-test/LOCAL_CLUSTER_RUNBOOK.md` for that.
* Auto-detect the right model size for your hardware — pick by
  hand based on (per-Mac memory) × (number of Macs) × 0.6.
* Cryptographically sign the model weights or the manifest;
  CoordPy's `META_MANIFEST` (Theorem W3-36) is content-addressed,
  not signed.

---

*This runbook is the operator boundary. The CoordPy-side
integration is one class
(`MLXDistributedBackend`) and one optional kwarg on
`run_sweep`. CoordPy deliberately does not absorb cluster
bring-up; the CoordPy contract is product, the cluster is
infrastructure.*
