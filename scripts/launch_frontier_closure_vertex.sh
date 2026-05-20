#!/usr/bin/env bash
#
# W86 frontier closure — Vertex AI Colab Enterprise orchestrator.
#
# Bundles the current repo (no GitHub push needed), uploads to
# GCS, ensures the HF token secret is in Secret Manager, creates
# (or reuses) the L4 runtime template, and submits the notebook
# execution. Designed to be re-runnable.
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/launch_frontier_closure_vertex.sh
#   (HF_TOKEN env var only used to populate / rotate the Secret
#   Manager secret on the FIRST run; subsequent runs read from
#   Secret Manager directly.)

set -euo pipefail

PROJECT="${PROJECT:-gen-lang-client-0387794233}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-gs://coordpy-frontier-closure-1779235853}"
SECRET_ID="${SECRET_ID:-coordpy-hf-token}"
TEMPLATE_NAME="${TEMPLATE_NAME:-coordpy-l4-bf16}"
DISPLAY_NAME_PREFIX="${DISPLAY_NAME_PREFIX:-w86-frontier-closure}"
NOTEBOOK_PATH="${NOTEBOOK_PATH:-scripts/colab_frontier_closure_w86.ipynb}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"

echo "[launch] PROJECT=$PROJECT  REGION=$REGION  BUCKET=$BUCKET"

# 1. Bundle the repo (exclude .venv, build artefacts, .git).
TS="$(date -u +%Y%m%dT%H%M%SZ)"
TAR_LOCAL="/tmp/coordpy_repo_${TS}.tar.gz"
echo "[launch] bundling repo -> $TAR_LOCAL"
tar --exclude='.venv' \
    --exclude='.release-venv' \
    --exclude='.git' \
    --exclude='.pytest_cache' \
    --exclude='.hypothesis' \
    --exclude='build' \
    --exclude='dist' \
    --exclude='*.egg-info' \
    --exclude='node_modules' \
    --exclude='__pycache__' \
    --exclude='results' \
    -czf "$TAR_LOCAL" \
    -C "$(dirname "$REPO_ROOT")" \
    "$(basename "$REPO_ROOT")"

# 2. Upload bundle + notebook to GCS.
gcloud storage cp "$TAR_LOCAL" "$BUCKET/repo_snapshot.tar.gz" \
    --project="$PROJECT"
gcloud storage cp "$NOTEBOOK_PATH" \
    "$BUCKET/notebooks/$(basename "$NOTEBOOK_PATH")" \
    --project="$PROJECT"

# 3. Ensure HF token secret exists. If HF_TOKEN env var is set,
#    create or add a new version of the secret.
if [[ -n "${HF_TOKEN:-}" ]]; then
    if gcloud secrets describe "$SECRET_ID" \
            --project="$PROJECT" >/dev/null 2>&1; then
        echo "[launch] adding new version to existing secret $SECRET_ID"
        printf '%s' "$HF_TOKEN" | gcloud secrets versions add \
            "$SECRET_ID" --project="$PROJECT" --data-file=-
    else
        echo "[launch] creating new secret $SECRET_ID"
        printf '%s' "$HF_TOKEN" | gcloud secrets create \
            "$SECRET_ID" --project="$PROJECT" \
            --replication-policy=automatic --data-file=-
    fi
    NOTEBOOK_SA_PROJECT_NUMBER="$(gcloud projects describe \
        "$PROJECT" --format='value(projectNumber)')"
    SA="${NOTEBOOK_SA_PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    gcloud secrets add-iam-policy-binding "$SECRET_ID" \
        --project="$PROJECT" \
        --member="serviceAccount:$SA" \
        --role="roles/secretmanager.secretAccessor" \
        --quiet 2>&1 | tail -5 || true
else
    echo "[launch] HF_TOKEN env not set; assuming secret $SECRET_ID is already present"
    gcloud secrets describe "$SECRET_ID" --project="$PROJECT" \
        >/dev/null 2>&1 \
        || { echo "ERROR: secret $SECRET_ID missing and HF_TOKEN not set"; exit 1; }
fi

# 4. Ensure runtime template exists.
TEMPLATE_ID="$(gcloud colab runtime-templates list \
    --region="$REGION" --project="$PROJECT" \
    --filter="displayName=$TEMPLATE_NAME" \
    --format="value(name.basename())" 2>/dev/null | head -1)"
if [[ -z "$TEMPLATE_ID" ]]; then
    echo "[launch] creating runtime template $TEMPLATE_NAME (L4)"
    gcloud colab runtime-templates create \
        --project="$PROJECT" --region="$REGION" \
        --display-name="$TEMPLATE_NAME" \
        --description="L4 GPU for CoordPy frontier closure runs" \
        --machine-type=g2-standard-12 \
        --accelerator-type=NVIDIA_L4 \
        --accelerator-count=1 \
        --disk-size-gb=200 \
        --disk-type=PD_SSD \
        --idle-shutdown-timeout=1h
    TEMPLATE_ID="$(gcloud colab runtime-templates list \
        --region="$REGION" --project="$PROJECT" \
        --filter="displayName=$TEMPLATE_NAME" \
        --format="value(name.basename())" | head -1)"
else
    echo "[launch] reusing runtime template $TEMPLATE_NAME ($TEMPLATE_ID)"
fi

# 5. Submit the execution.
EXEC_NAME="${DISPLAY_NAME_PREFIX}-${TS}"
echo "[launch] submitting execution $EXEC_NAME"
gcloud colab executions create \
    --project="$PROJECT" --region="$REGION" \
    --display-name="$EXEC_NAME" \
    --notebook-runtime-template="$TEMPLATE_ID" \
    --gcs-notebook-uri="$BUCKET/notebooks/$(basename "$NOTEBOOK_PATH")" \
    --gcs-output-uri="$BUCKET/executions/" \
    --execution-timeout=4h

# 6. List recent executions so the caller knows what to monitor.
echo "[launch] recent executions:"
gcloud colab executions list \
    --project="$PROJECT" --region="$REGION" \
    --sort-by=createTime \
    --format="table(displayName,jobState,createTime)" \
    --limit=5

echo "[launch] DONE. Bundle uploaded to $BUCKET/repo_snapshot.tar.gz"
echo "[launch] Notebook uploaded to $BUCKET/notebooks/$(basename "$NOTEBOOK_PATH")"
echo "[launch] Outputs will appear at $BUCKET/w86_<TS>/ after the run."
