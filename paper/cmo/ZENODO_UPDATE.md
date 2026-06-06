# Updating the Zenodo deposit for the CMO paper

This is the operator runbook for refreshing the Zenodo archive after a round of
manuscript/figure changes. The **packaging is automated; the publish step is
manual** (it mints a DOI and is irreversible — keep it human).

## The two records

| Record | Role | In-repo references |
|---|---|---|
| **Paper archive** — concept DOI `10.5281/zenodo.20444215` | Self-contained bundle: manuscript, figures, tables, audit, reproducibility scripts, key data. Rebuilds the PDF and every figure from the bundle alone. | `manuscript.tex` (Data availability + Reproducibility Statement), `rebuild_from_zenodo.sh` |
| **Heavy specimen data** — `10.5281/zenodo.20369312` | The five `specimen_*.npz` dense reconstructions (~120 MB). | `examples/zenodo_fetch.py`, `CHANGELOG.md`, `docs/RELEASE_READINESS.md` |

> **Heads-up (2026-06):** the *five-variant* dense figure was removed from the
> manuscript, so **no paper artefact depends on `20369312` anymore**. Updating it
> is optional — refresh it for repository consistency, or retire it. The paper
> archive (`20444215`) is the one that matters.

### DOI references to reconcile (manual, owner only)

The repo currently cites several version DOIs inconsistently — reconcile these
against the live Zenodo account before/after publishing the new version:

- `manuscript.tex` cites concept `20444215` + submitted **v4 `20533009`**.
- `paper/cmo/SUBMISSION_CHECKLIST.md` and `docs/VALIDATION_STATUS.md` cite
  `20444216`.
- the old `rebuild_from_zenodo.sh` history referenced `20444786`.

Decide the canonical concept DOI, then make every reference point to it (concept
for landing, version DOI only where exact reproducibility is promised).

## Procedure

### 1. Build the bundle (automated)

```bash
cd paper/cmo && make pdf            # ensure manuscript.pdf is current
rtk .venv/bin/python paper/cmo/make_zenodo_bundle.py
rtk .venv/bin/python paper/cmo/make_zenodo_bundle.py --verify
```

Produces `paper/cmo/build/cmo_paper_bundle.zip` (structured, preserves repo
layout) + `BUNDLE_MANIFEST.json` (sha256 of every file). The zip lives under the
git-ignored `build/`, so it is never committed.

### 2. Dry-run, then sandbox

```bash
# see the plan without contacting Zenodo
rtk .venv/bin/python examples/zenodo_upload.py --record <latest_version_id> \
    --replace --version v5 \
    --files paper/cmo/build/cmo_paper_bundle.zip paper/cmo/manuscript.pdf --dry-run

# rehearse against sandbox.zenodo.org (needs a sandbox token)
ZENODO_TOKEN=<sandbox> rtk .venv/bin/python examples/zenodo_upload.py --sandbox \
    --record <sandbox_record> --replace --version v5 \
    --files paper/cmo/build/cmo_paper_bundle.zip paper/cmo/manuscript.pdf
```

`--record` is the numeric id of the **latest published version**, not the
concept-DOI id.

### 3. Create the new version on production (no publish yet)

```bash
ZENODO_TOKEN=<prod> rtk .venv/bin/python examples/zenodo_upload.py \
    --record <latest_version_id> --replace --version v5 \
    --files paper/cmo/build/cmo_paper_bundle.zip paper/cmo/manuscript.pdf
```

It prints the **reserved DOI** and the draft URL. The draft is editable; nothing
is public yet.

### 4. Finalize the in-repo references with the reserved DOI

Before publishing, freeze these into the manuscript so the archived PDF is
self-consistent, then rebuild and re-bundle:

- `manuscript.tex` Data-availability + Reproducibility Statement: set the new
  version DOI (replaces `20533009`) and the **current commit hash** (replaces
  `08d1b25` on line ~996; use `git rev-parse --short HEAD`).
- re-run steps 1 and 3 so the bundle carries the finalized PDF (or upload the
  rebuilt `manuscript.pdf` to the same draft).

### 5. Publish (manual, irreversible)

Review the draft in the browser, then either click **Publish** on Zenodo or:

```bash
ZENODO_TOKEN=<prod> rtk .venv/bin/python examples/zenodo_upload.py \
    --record <latest_version_id> --files <...> --publish
```

### 6. (Optional) refresh the heavy data record

```bash
ZENODO_TOKEN=<prod> rtk .venv/bin/python examples/zenodo_upload.py \
    --record 20369312 --replace \
    --files docs/assets/pycaso_real_data/schur_ba/specimen_*.npz
```

## Verifying a published archive

```bash
# in a scratch dir, from the published bundle
unzip cmo_paper_bundle.zip
bash rebuild_from_zenodo.sh          # checks every file against BUNDLE_MANIFEST.json
cd paper/cmo && make repro           # rebuilds the PDF + audits every number
```
