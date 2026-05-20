# Colab Pro — CoordPy Frontier Closure Runbook

> User-driven runbook for executing CoordPy GPU-bound benches on
> a Google **Colab Pro browser subscription**
> (`colab.research.google.com`). Used in W86 to close the
> post-W83 P0 substrate blockers (#25 / #26 / #27 hidden-state-
> intercept axis). Manual: the user opens the notebook in a
> browser and clicks Run-All.
>
> Note: Colab Pro does NOT have a programmatic CLI. The
> separate `gcloud colab …` CLI is *Colab Enterprise on Vertex
> AI*, which bills the user's GCP project and is OFF-LIMITS for
> this project per user preference. See
> `memory/feedback_no_gcp_charges_use_colab_pro_browser.md`.

## One-time setup (per Google account)

1. **Subscribe** to Colab Pro at https://colab.research.google.com/signup if not already.
2. **Accept Meta Llama-3.1 license** at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct — the page should say "you have been granted access".
3. **Mint a HuggingFace read token** at https://huggingface.co/settings/tokens (role: Read; name: something like `coordpy-frontier-closure`).

## Per-run procedure (~30–60 min wall-clock)

1. Open the notebook in Colab:
   - Direct URL: `https://colab.research.google.com/github/adotdong29/CoordPy/blob/main/scripts/colab_frontier_closure_w86.ipynb`
   - (Or upload `scripts/colab_frontier_closure_w86.ipynb` via *File → Upload notebook*.)
2. **Select GPU runtime:** *Runtime → Change runtime type → A100 / V100 / L4 / T4 GPU*.
   - A100 (40 GB) is best. ~30 min total wall-clock.
   - V100 / L4 (16–24 GB) work too. ~60 min.
   - T4 (16 GB) will OOM at the 32 k token bar; lower `--horizon-tokens 8192` and the #27 bar becomes "≥ 8k" rather than "≥ 32k".
3. **Set the HF token secret:**
   - Click the **🔑 key icon** in the left sidebar.
   - *+ Add new secret* → name = `hf_token`, value = `hf_xxxxxxxx`.
   - Toggle *Notebook access* **on**.
4. **Run-All:** *Runtime → Run all*.
5. **Mount Drive when prompted** (cell 8 will trigger an OAuth popup). This is the only interactive step; it saves results to `/content/drive/MyDrive/coordpy_frontier_closure/w86_<TS>/`.
6. **Download the zip** that the final cell offers (backup in case Drive sync fails).

## After the run

* Share the Drive folder back with Andy, or attach the downloaded zip to a chat message.
* The audit chain re-verifies offline:
  ```bash
  python scripts/verify_w86_audit_chain.py \
      --report results/w86/<TS>/frontier_closure_report.json
  ```

## What the run produces

* `frontier_closure_report.json` — top-level content-addressed report.
* `25_substrate_coupling.json` — W80 conformance + hidden-state intercept + replay-from-KV at frontier scale.
* `26_live_learned_memory.json` — live-vs-synthetic MSE strict-beat verdict.
* `27_long_context_intercept.json` — hidden-state intercept moves CID at ≥ 32 k tokens.

Every report carries a `report_cid` that the verifier re-hashes
and compares against the recorded value. The audit chain is
offline-re-verifiable from disk.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `userdata.SecretNotFoundError: hf_token` | The Colab Secret wasn't created or notebook access toggle is off. Re-do step 3. |
| `OSError: You are trying to access a gated repo` | Llama-3.1-8B license not accepted on your HF account. Accept it at huggingface.co/meta-llama/Llama-3.1-8B-Instruct and re-run. |
| `CUDA out of memory` at 32 k tokens | The runtime gave you a smaller GPU than expected. Re-attempt with A100 (Runtime → Change runtime type), OR lower `--horizon-tokens` in cell 5 to 8192. The #25 / #26 bars still pass. |
| `Connection lost; reconnect?` | Colab Pro can drop sessions on long runs. Output cells already wrote to Drive in cell 8, so progress is preserved. Re-run from cell 8 onwards. |
| `git clone` hangs | Public GitHub clone occasionally throttles; retry. If persistent, use `https://github.com/adotdong29/CoordPy/archive/main.tar.gz` instead. |

## Cost

Zero additional cost — Colab Pro is a flat monthly subscription, no per-run billing.
