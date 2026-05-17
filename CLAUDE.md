# StereoComplex — project summary (2026-05-17)

## Project purpose

StereoComplex is a Python library for non-central stereo camera calibration via
Zernike rayfields. Each pixel gets its own 3D ray (origin + direction), replacing
the single-pinhole abstraction with a per-pixel line model. It identifies optical
architectures (pinhole, Brown-Conrady, inclined plate, CMO, Greenough) through
BIC-based model selection on the fitted rayfield.

## Active tasks

### 11_compare_zernike_cmo_rigid_removed.py
Compare Zernike 57p vs CMO 26p specimen reconstructions after rigid-body removal.
Goal: determine if the Zernike ramp is a global frame effect, a per-arm alignment difference, or a genuine non-rigid rayfield component.
8 phases — see CDC in session.

## Key files and directories

| Path | Role |
|---|---|
| `src/stereocomplex/` | Library source (physics/, rayfields/, benchmarks/) |
| `tests/` | 109 tests pass (39 slow deselected), ~15 s fast mode |
| `examples/notebooks/` | 10 notebooks (00-09, plus 10 specimen reconstruction) |
| `paper/cmo/` | CMO journal paper (manuscript.tex, references.bib, 10 figures, 5 tables) |
| `paper/cmo/build/` | Compiled PDF (740-line manuscript, 24 pages) |
| `paper/cmo/figures/` | 10 PNG figures at 200 dpi |
| `paper/cmo/tables/` | 5 auto-generated booktabs LaTeX tables |
| `paper/cmo/generate_tables.py` | Reads JSON artifacts → LaTeX tables |
| `paper/cmo/check_manuscript_numbers.py` | 27 audit checks against JSON (all OK) |
| `paper/cmo/SUBMISSION_CHECKLIST.md` | Pre-submission checklist |
| `paper/cmo/number_audit_report.md` | Full number audit report |
| `docs/REAL_CMO_PYCASO_RAYFIELD.md` | Pedagogical case study (7 steps) |
| `docs/assets/pycaso_real_data/` | 45 JSON/NPZ/SVG/PNG artifacts (R3XA steps, sweeps, diagnostics) |

## CMO paper status

**v1.0-submission-ready.** 24 pages, 44 references, 10 figures, 5 tables.

Core narrative: double TPS denoising → Zernike rayfield BA → BIC model selection
→ CMO+SE(3) 26p physical model at 1.06 px reprojection on real Pycaso data.
Specimen reconstruction sanity check: DIS optical flow on independent coin pair,
2M correspondences, 0.107 mm median ray gap, confirms rayfield coherence.

Remaining human tasks: fill Zenodo DOI, confirm contact email, optionally contact
Pycaso authors for baseline/WD nominal values.

### Key results (all audit-verified against JSON artifacts)

- 26p CMO+SE(3) model: 1.06 px RMS reprojection on 165 ChArUco corners
- Ray-space BIC selects 14p (compressed), operational BIC with 1.5 px guard selects 26p
- SE(3) arm alignment reduces ray RMS from 0.16→0.002 mm (98% reduction)
- Schur coupling norm c=0.81 (pose/ray coupling, not a conditioning problem)
- Z0 dominates residual (97-98%) — gauge mode, not physics error
- Specimen: CMO Z MAD=0.073 mm, Zernike Z MAD=0.194 mm (CMO smoother)

### Paper figures (paper/cmo/figures/)

1. `pipeline.png` — Methodological pipeline diagram
2. `cmo_physical.png` — CMO geometry with sub-pupil paths
3. `bic_bars.png` — BIC bar chart (ray-space + operational)
4. `BIC_vs_order.png` — BIC vs Zernike order sweep
5. `dy_profile_comparison.png` — dy profile: Zernike vs CMO vs residual
6. `residual_evolution.png` — Residual evolution through the 7 steps
7. `subpupil_3d.png` — 3D sub-pupil positions
8. `schur_singular_values.png` — Schur complement singular values
9. `pareto_gauge_regularization.png` — Pareto frontier gauge regularization
10. `specimen_reconstruction.png` — 2×3 specimen reconstruction panel

## Recent accomplishments (last 15 commits)

- Manuscript complete: Introduction through Reproducibility Statement
- 7 explicit limitation points in dedicated Limitations section
- Number audit: 27/27 checks pass
- Auto-generated LaTeX tables from JSON artifacts
- Specimen reconstruction sanity check (notebook 10, DIS ULTRAFAST flow)
- R3XA reproducibility metadata (4 steps with SVG graphs)
- CMOWarpedStereoModel implemented (polynomial pre-warp layer, ~350 lines)
- Double TPS denoising producing gauge-stable rayfield

## Active conventions

### Deep-claude workflow
Always use `--resume c7c56802-1828-4013-a380-be256e554caa` for deep-claude sessions.
The session transcript is at `~/.claude/projects/-home-jeff-StereoComplex/c7c56802...`.

### Permissions
Use `--dangerously-skip-permissions` for batch operations in this repo.
The user prefers minimal permission prompts during sustained work sessions.

### Code quality
- No half-finished implementations, no dead code, no commented-out blocks
- Three similar lines > premature abstraction
- No backwards-compat shims or deprecation aliases (v0.x is explicitly unstable)
- 109 tests must always pass
- `filterwarnings = ["error::DeprecationWarning"]` in pyproject.toml

### Paper workflow
- Build PDF: `cd paper/cmo && bash build_pdflatex.sh`
- Number audit: `python3 paper/cmo/check_manuscript_numbers.py`
- Generate tables: `python3 paper/cmo/generate_tables.py`
- Paper CLAUDE.md at `paper/cmo/CLAUDE.md` has 6-phase polish plan (all complete)

### Specimen reconstruction
- Uses Pycaso `left_identification/coin.tif` and `right_identification/coin.tif`
- DIS ULTRAFAST flow: finestScale=0, patchSize=8, iterations=20, no variational refinement
- ROI: [300:W-300, 300:H-300], 1448×1448 effective region
- Both U and V flow components (images NOT rectified)
- Triangulation with both CMO 26p and Zernike 57p rayfields
- Figure auto-scales ray gap histogram xlim to 99.5th percentile
