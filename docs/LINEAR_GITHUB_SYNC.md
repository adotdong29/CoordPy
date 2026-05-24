# Linear ↔ GitHub sync discipline

Linear (workspace [coordpy](https://linear.app/coordpy/)) is the
second brain for this programme.  GitHub is the canonical source
of truth for code, results, and docs.  This page is the
operating contract that keeps them in sync without becoming
ceremony.

## What lives where

| Surface | Authority | Notes |
|---|---|---|
| Code, results JSON, sidecars, runbooks, RESULTS docs | **GitHub** (this repo) | Linear never duplicates code — it references it. |
| Active queue (what needs doing, what just landed, what got killed) | **Linear** | One obvious current queue. |
| Carry-forwards (`W##-L-*-CAP` registry + retirement bars) | **GitHub** (`docs/THEOREM_REGISTRY.md`, `docs/RESEARCH_STATUS.md`) | Linear references them; never restates them. |
| Per-milestone audit chain (Merkle roots, SHAs) | **GitHub** (`results/wXX/...`) | Linear comments link to specific run dirs. |
| Project narrative + truth surface (post-WNN) | **Linear documents** (`coordpy` project docs) | Updated end-of-milestone. |

## Operating rules

1. **Update GitHub first.**  Code, results JSON, runbooks, and
   RESULTS docs land in the repo before Linear.
2. **Sync Linear at end-of-milestone, not mid-stream.**  Issues
   should reflect the latest *landed* state, not a running
   diary.  A milestone is "done" when its RESULTS doc is
   committed.
3. **Comment with links, not duplicates.**  Linear comments
   should reference repo paths and commit SHAs, not paste
   prose.
4. **Mark dead directions dead.**  If a Linear issue's
   hypothesis is empirically retired, set status to Canceled or
   Done with a comment naming the carry-forward.  Do not let
   stale issues sit in Todo.
5. **Use `linear_github_mapping.json`** as the canonical bridge:
   for every milestone, list the commits, the docs, the Linear
   issue IDs, and the carry-forwards.
6. **Validate before commenting**.  Run `python
   scripts/sync_linear_github_v1.py validate` to confirm every
   referenced commit + doc actually exists and every commit is
   pushed to origin.
7. **One snapshot per major sync**: run `python
   scripts/sync_linear_github_v1.py snapshot --milestone W##`
   and paste the output as the W## end-of-milestone comment on
   the parent issue.
8. **Re-runs are cheap.**  The sync helper is read-only +
   idempotent.  Failure-mode: it prints `DRIFT:` lines and
   exits non-zero so CI / future agents can detect when GitHub
   and Linear are out of step.

## Canonical mapping data

The file `linear_github_mapping.json` at repo root is the
single source of truth for the GitHub ↔ Linear bridge.  Its
schema:

```json
{
  "schema": "coordpy.linear_github_mapping.v1",
  "team": "CoordPy",
  "project": "CoordPy",
  "linear_workspace_url": "https://linear.app/coordpy/",
  "github_repo": "adotdong29/CoordPy",
  "milestones": [
    {
      "id": "W95",
      "title": "...",
      "outcome": "...",
      "commits": ["<sha>"],
      "docs": ["docs/RUNBOOK_W95.md", "..."],
      "linear_issues": ["COO-11", "COO-16"],
      "carry_forwards_retired": [],
      "carry_forwards_added": []
    }
  ]
}
```

### Adding a milestone

1. Land code + commits in `main`.
2. Write the RESULTS doc.
3. Append a milestone entry to `linear_github_mapping.json`.
4. Run `python scripts/sync_linear_github_v1.py validate`.
   Fix any drift before continuing.
5. Run `python scripts/sync_linear_github_v1.py snapshot
   --milestone W##` and paste the output as a comment on the
   parent Linear issue (`COO-6` for the post-W93 backlog).
6. Update the relevant child Linear issues (set status, drop
   stale fields).

### When NOT to sync

* Don't sync mid-WIP.  Wait until the milestone has a RESULTS
  doc.
* Don't sync to Linear what's already in CHANGELOG.md or
  THEOREM_REGISTRY.md — Linear should *link*, not duplicate.
* Don't sync a candidate that hasn't passed preflight to Linear
  as "active".  Use the Linear backlog (status `Backlog`) for
  preflight-pending ideas; promote to `Todo` / `In Progress`
  only once preflight passes.

## End-of-W95 sync recipe (worked example)

```bash
# 1) verify the GitHub side
python scripts/sync_linear_github_v1.py validate

# 2) preview the Linear comment markdown
python scripts/sync_linear_github_v1.py snapshot --milestone W95

# 3) using the Linear MCP, set:
#    - COO-11 description = "Build MathVista harness (W95) — preflight earned Phase 2"
#    - COO-16 add comment with the snapshot output
#    - COO-6 add comment "W95 preflight: PASS; pilot entitled but not launched"
```

## Why this stays durable

* The mapping is **versioned with the code**.  Linear-state
  drift is detectable from the repo alone.
* The script is **read-only**.  No accidental Linear mutations.
* Linear writes remain **explicit MCP calls**, which leaves an
  audit trail in conversation history.
* The schema is **append-only**.  Old milestone entries are
  never edited in place; they document what was true at that
  milestone's close.
