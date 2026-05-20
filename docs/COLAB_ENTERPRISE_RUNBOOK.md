# Vertex AI Colab Enterprise — CoordPy Frontier Closure Runbook

> Runbook for executing CoordPy GPU-bound benches and closure
> runs on Google Cloud's *Colab Enterprise* product
> (managed Vertex AI Workbench Notebook Executions, fronted by
> the `gcloud colab …` CLI). Distinct from the consumer
> `colab.research.google.com` Pro subscription, which has no
> programmatic CLI. Used in W86 to close the post-W83 P0 substrate
> blockers (#25 / #26 / #27 hidden-state-intercept axis).

## Why Colab Enterprise (not Colab Pro)?

`gcloud colab` is the only Google-supported CLI for
programmatically executing notebooks. It runs on Vertex AI
Workbench backends, is billed per GCP rules (per-second metered
machine + GPU + storage), and integrates with Secret Manager,
GCS, IAM, and the rest of Cloud. Standard Colab Pro
(consumer subscription on `colab.research.google.com`) has no
public CLI for batch notebook execution; using it requires a
human in a browser, which we cannot drive from here.

## Cost ballpark (us-central1, on-demand list price, 2026-05)

| Machine / GPU              | $ / hr (machine + GPU) |
|----------------------------|------------------------|
| `g2-standard-12` + 1 × L4  | ~$0.85                 |
| `a2-highgpu-1g` + 1 × A100-40| ~$3.67               |
| `a2-ultragpu-1g` + 1 × A100-80 | ~$5.07             |

For W86 we target the L4 because (a) project quota was 1 × L4
and 0 × A100, (b) Llama-3.1-8B-Instruct fits in 24 GB at bf16,
(c) cost is in the single-digit dollars for the whole run.

## Prerequisites (one-time per GCP project)

```bash
PROJECT=gen-lang-client-0387794233
REGION=us-central1

# Enable APIs.
gcloud services enable aiplatform.googleapis.com --project=$PROJECT
gcloud services enable storage.googleapis.com --project=$PROJECT
gcloud services enable notebooks.googleapis.com --project=$PROJECT
gcloud services enable secretmanager.googleapis.com --project=$PROJECT

# Verify billing is enabled (Colab Enterprise requires it).
gcloud billing projects describe $PROJECT

# Check GPU quota — A100 quota is usually 0 on a new project;
# L4 is generally 1 by default. Quota increase requests go through
# console.cloud.google.com/iam-admin/quotas .
gcloud compute regions describe $REGION --project=$PROJECT \
  --format=json | python3 -c "
import json,sys
d=json.load(sys.stdin)
for q in d.get('quotas',[]):
    if any(k in q['metric'] for k in (
            'NVIDIA_A100','NVIDIA_L4','NVIDIA_V100','NVIDIA_T4')):
        print(f'{q[\"metric\"]:38} limit={q[\"limit\"]} usage={q.get(\"usage\",0)}')
"

# Make a GCS bucket for outputs.
BUCKET=gs://coordpy-frontier-closure-$(date +%s)
gcloud storage buckets create $BUCKET \
  --location=$REGION --project=$PROJECT \
  --uniform-bucket-level-access
echo "BUCKET=$BUCKET"
```

## Secret Manager: store the HF token

```bash
# 1. Mint a HF read token at https://huggingface.co/settings/tokens
# 2. Accept the model license at the gated repo page (e.g.,
#    https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct ).
# 3. Store the token in Secret Manager:
printf 'hf_XXXXXXXX' | gcloud secrets create coordpy-hf-token \
  --project=$PROJECT --data-file=-

# 4. Grant the Notebook Execution service account read access:
NOTEBOOK_SA=$(gcloud projects describe $PROJECT --format='value(projectNumber)')
NOTEBOOK_SA="${NOTEBOOK_SA}-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding coordpy-hf-token \
  --project=$PROJECT \
  --member="serviceAccount:${NOTEBOOK_SA}" \
  --role="roles/secretmanager.secretAccessor"
```

## Create a runtime template (L4 GPU)

```bash
gcloud colab runtime-templates create \
  --project=$PROJECT --region=$REGION \
  --display-name=coordpy-l4-bf16 \
  --description="L4 GPU for CoordPy frontier closure runs" \
  --machine-type=g2-standard-12 \
  --accelerator-type=NVIDIA_L4 \
  --accelerator-count=1 \
  --disk-size-gb=200 \
  --disk-type=PD_SSD \
  --idle-shutdown-timeout=1h

# Confirm:
gcloud colab runtime-templates list --region=$REGION --project=$PROJECT
TEMPLATE_ID=$(gcloud colab runtime-templates list \
  --region=$REGION --project=$PROJECT \
  --filter="displayName=coordpy-l4-bf16" \
  --format="value(name.basename())")
echo "TEMPLATE_ID=$TEMPLATE_ID"
```

## Submit a notebook execution

```bash
# Upload the notebook to GCS so Colab Enterprise can fetch it.
gcloud storage cp scripts/colab_frontier_closure_w86.ipynb \
  $BUCKET/notebooks/colab_frontier_closure_w86.ipynb \
  --project=$PROJECT

# Submit.
gcloud colab executions create \
  --project=$PROJECT --region=$REGION \
  --display-name=w86-frontier-closure-$(date +%s) \
  --notebook-runtime-template=$TEMPLATE_ID \
  --gcs-notebook-uri=$BUCKET/notebooks/colab_frontier_closure_w86.ipynb \
  --gcs-output-uri=$BUCKET/executions/ \
  --execution-timeout=4h
```

## Monitor + collect

```bash
# List recent executions and their state:
gcloud colab executions list --project=$PROJECT --region=$REGION \
  --sort-by=createTime --limit=10

# Tail a specific execution:
EXEC_ID=...
gcloud colab executions describe $EXEC_ID \
  --project=$PROJECT --region=$REGION
gcloud colab executions logs $EXEC_ID \
  --project=$PROJECT --region=$REGION

# Pull results once SUCCEEDED:
gcloud storage cp -r $BUCKET/w86_<RUN_TS>/ ./results/w86/
```

## Re-verifying the closure offline

The driver writes a content-addressed
`frontier_closure_report.json` plus per-issue sidecar JSON
files. Re-verification:

```bash
python scripts/verify_w86_audit_chain.py \
  --report results/w86/<RUN_TS>/frontier_closure_report.json
```

(See `scripts/verify_w86_audit_chain.py` for the per-issue CID
checks.)

## Why not gcloud aiplatform custom-jobs?

Vertex AI custom training jobs (`gcloud ai custom-jobs create`)
are an older surface. Colab Enterprise notebook executions are
the supported, batch-friendly successor: same backend (Vertex
AI Workbench), simpler config (just point at a `.ipynb` on
GCS), and the runtime template captures GPU + machine spec as
a reusable handle.

## Tearing down

```bash
# Delete runtime template (does not affect billing for past runs).
gcloud colab runtime-templates delete $TEMPLATE_ID \
  --project=$PROJECT --region=$REGION

# Delete secret if you rotate the HF token.
gcloud secrets delete coordpy-hf-token --project=$PROJECT

# Delete GCS bucket (irreversible; only after pulling results).
gcloud storage rm -r $BUCKET --project=$PROJECT
```

## What to do when something fails

1. **Quota error**: open
   `https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT`,
   request an increase on the specific GPU SKU. Approvals are
   sometimes minutes, sometimes days.
2. **Secret not found**: re-grant the IAM binding (see above).
3. **OOM on L4**: the closure driver supports `--horizon-tokens
   8192` as a fallback below 32 k; the #27 bar is NOT met but
   #25 / #26 still are.
4. **HF gated model 401**: the user's HF account has not
   accepted the Meta Llama-3.1 license on the gated repo page;
   accept and re-run.
5. **Notebook stalls**: `execution-timeout` defaults are
   conservative; set `--execution-timeout=4h` explicitly.
