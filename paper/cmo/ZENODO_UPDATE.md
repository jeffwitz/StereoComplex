# Updating the Zenodo deposit for the CMO paper

This is the operator runbook for refreshing the Zenodo archive after a round of
manuscript/figure changes. The **packaging is automated; the publish step is
manual** (it mints a DOI and is irreversible — keep it human).

## The two records

| Record | Concept DOI | Latest published version | Role |
|---|---|---|---|
| **Paper archive** | `10.5281/zenodo.20444215` | **v5 = `20574710`** (record id `20574710`, published 2026-06-06) | Self-contained bundle: manuscript, figures, tables, audit, scripts, key data. Rebuilds the PDF and every figure from the bundle alone. |
| **Heavy specimen data** | `10.5281/zenodo.20369311` | `20369312` | The five `specimen_*.npz` dense reconstructions (~120 MB). |

So the next paper-archive version (v6) is created with `--record 20574710`.

> **Heads-up (2026-06):** the *five-variant* dense figure was removed from the
> manuscript, so **no paper artefact depends on `20369312` anymore**. Updating it
> is optional — refresh it for repository consistency, or retire it. The paper
> archive (`20444215`) is the one that matters.

### Canonical DOI map (resolved 2026-06-06 from the public Zenodo API)

The paper-archive concept is **`20444215`**; its version chain is
`20444216` (90 files, 2026-05-29) → `20444786` (100 files, 2026-06-01) →
`20533009` = v4 (2026-06-03) →
**`20574710` = v5 (structured bundle + standalone PDF, 2026-06-06), the latest
published version**. All ids except `20574710` are *superseded versions* — never
cite them as the landing page.

Convention applied in-repo: cite the **concept DOI `20444215`** for the stable
landing, and a **version DOI only** where exact reproducibility is promised
(`manuscript.tex` pins the submitted version). Stale `20444216` references in
`SUBMISSION_CHECKLIST.md` and `docs/VALIDATION_STATUS.md` were corrected to the
concept DOI on 2026-06-06.

## Procedure

The manuscript must cite the *new* version DOI, so reserve the DOI **before**
building the final bundle. `--record` is always the numeric id of the **latest
published version** (currently `20574710`), not the concept-DOI id. The examples
below create v6 on top of v5.

### 1. Reserve the next version DOI (no files yet)

```bash
ZENODO_TOKEN=<prod> rtk .venv/bin/python examples/zenodo_upload.py \
    --record 20574710 --reserve-only
```

Prints the **reserved DOI** (e.g. `10.5281/zenodo.XXXXXXXX`) and the **draft
deposition id** — note both. Nothing is public; the draft is discardable.

### 2. Freeze the reserved DOI + commit hash into the manuscript

In `manuscript.tex` (Data-availability §982 + Reproducibility Statement §999):
set the new version DOI (replaces the previous version DOI) and bump the version
label (`v5` → `v6`); on line ~996 set the **current commit hash**
(`git rev-parse --short HEAD`). Commit.

### 3. Build the bundle (automated, PDF auto-synced)

```bash
cd paper/cmo && make bundle    # runs build_pdflatex.sh (which now copies
                               # build/manuscript.pdf -> paper/cmo/manuscript.pdf),
                               # then make_zenodo_bundle.py + --verify
```

Produces `paper/cmo/build/cmo_paper_bundle.zip` (structured, preserves repo
layout) + `BUNDLE_MANIFEST.json` (sha256 of every file). The zip lives under the
git-ignored `build/`, so it is never committed. The standalone
`paper/cmo/manuscript.pdf` is kept in sync by `build_pdflatex.sh` — no manual
copy needed.

### 4. Upload into the reserved draft (no publish yet)

`newversion` fails if a draft already exists, so target the draft directly with
`--draft-id <id from step 1>`:

```bash
ZENODO_TOKEN=<prod> rtk .venv/bin/python examples/zenodo_upload.py \
    --record 20574710 --draft-id <draft_id> --replace --version v6 \
    --files paper/cmo/build/cmo_paper_bundle.zip paper/cmo/manuscript.pdf
```

(`--dry-run` first to preview; `--sandbox --record <sandbox_record>` to rehearse
against sandbox.zenodo.org with a sandbox token.)

### 5. Publish (manual, irreversible)

Review the draft in the browser, then either click **Publish** on Zenodo or:

```bash
ZENODO_TOKEN=<prod> rtk .venv/bin/python examples/zenodo_upload.py \
    --record 20574710 --draft-id <draft_id> --publish
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
