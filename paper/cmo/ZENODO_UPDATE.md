# Updating the Zenodo deposit for the CMO paper

This is the operator runbook for refreshing the Zenodo archive after a round of
manuscript/figure changes. The **packaging is automated; the publish step is
manual** (it mints a DOI and is irreversible — keep it human).

## The two records

| Record | Concept DOI | Latest published version | Role |
|---|---|---|---|
| **Paper archive** | `10.5281/zenodo.20444215` | **v4 = `20533009`** (record id `20533009`) | Self-contained bundle: manuscript, figures, tables, audit, scripts, key data. Rebuilds the PDF and every figure from the bundle alone. |
| **Heavy specimen data** | `10.5281/zenodo.20369311` | `20369312` | The five `specimen_*.npz` dense reconstructions (~120 MB). |

So the next paper-archive version (v5) is created with `--record 20533009`.

> **Heads-up (2026-06):** the *five-variant* dense figure was removed from the
> manuscript, so **no paper artefact depends on `20369312` anymore**. Updating it
> is optional — refresh it for repository consistency, or retire it. The paper
> archive (`20444215`) is the one that matters.

### Canonical DOI map (resolved 2026-06-06 from the public Zenodo API)

The paper-archive concept is **`20444215`**; its version chain is
`20444216` (90 files, 2026-05-29) → `20444786` (100 files, 2026-06-01) →
**`20533009` = v4 (2026-06-03), the latest published version**. The bare
`20444216` / `20444786` ids are *superseded versions* — never cite them as the
landing page.

Convention applied in-repo: cite the **concept DOI `20444215`** for the stable
landing, and a **version DOI only** where exact reproducibility is promised
(`manuscript.tex` pins the submitted version). Stale `20444216` references in
`SUBMISSION_CHECKLIST.md` and `docs/VALIDATION_STATUS.md` were corrected to the
concept DOI on 2026-06-06.

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
rtk .venv/bin/python examples/zenodo_upload.py --record 20533009 \
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
    --record 20533009 --replace --version v5 \
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
    --record 20533009 --files <...> --publish
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
