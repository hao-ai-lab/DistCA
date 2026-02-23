# CI — Modal GPU Smoke Test

This CI workflow proves that DistCA compiles and runs on real GPU hardware by
executing the full "install + build csrc + single-GPU pretrain" pipeline on a
Modal H100.

## What it runs

1. **Planner unit tests** — `tests/test_planner.py` and `tests/test_items.py`
2. **Single-GPU smoke test** — `scripts/docker_install_and_build.sh --smoke`
   (installs distca, builds csrc/libas_comm, runs 1-batch pretrain_llama.py)

## Setup — GitHub Secrets

You need **two** repository secrets so the GitHub Actions runner can talk to
your Modal account.

### 1. Create a Modal API token

```bash
# Install Modal locally (if you haven't)
pip install modal

# Create a token — this opens the browser to log in
modal token new
```

After logging in, Modal stores the credentials in `~/.modal.toml`. You can
view them:

```bash
cat ~/.modal.toml
```

You'll see two values:
- `token_id`     → this becomes `MODAL_TOKEN_ID`
- `token_secret` → this becomes `MODAL_TOKEN_SECRET`

### 2. Add the secrets to GitHub

Go to your repo → **Settings** → **Secrets and variables** → **Actions** →
**New repository secret**:

| Secret name          | Value                              |
|---------------------|------------------------------------|
| `MODAL_TOKEN_ID`     | The `token_id` from `~/.modal.toml`  |
| `MODAL_TOKEN_SECRET` | The `token_secret` from `~/.modal.toml` |

Or use the CLI:

```bash
gh secret set MODAL_TOKEN_ID     --body "<your-token-id>"
gh secret set MODAL_TOKEN_SECRET --body "<your-token-secret>"
```

### 3. Verify

Push to `main` or open a PR — the **"CI — Modal Smoke Test"** workflow should
appear in the Actions tab.

## Running locally

```bash
# From repo root
modal run ci/modal_smoke_test.py
```

## Cost

Each run uses ~10-30 minutes of Modal H100 time (\~$0.33/min as of mid-2026).
The CI uses concurrency groups to cancel stale runs on force-pushes.
