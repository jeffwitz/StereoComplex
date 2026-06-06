# Validation status

| Component | Status | Evidence | Remaining work |
|---|---|---|---|
| Ray2D preprocessing | Synthetic validated | `docs/CHARUCO_IDENTIFICATION.md`, notebook 01 | Real noisy image benchmark |
| Non-central Zernike rayfield | Synthetic validated | Plate oracle, CMO oracle (notebook 04, 06) | Real microscope data |
| Physical model selection | Synthetic validated | 6-oracle matrix (`CMO_MODEL_SELECTION.md`) | Broader catalogue, real instruments |
| Physical CMO shared-rig | Synthetic validated | CMO oracle recovery (`CMO_PHYSICAL_MODEL.md`) | Real CMO microscope |
| Direct vs rayfield mediation | **Exploratory** | Notebook 08, direct expected-model fit | Symmetric multi-candidate comparison |
| Zernike compact fallback | Synthetic validated | Uncatalogued Zernike oracle (notebook 07) | Real unknown optics |

**Key:**

- **Synthetic validated** — demonstrated on synthetic oracles with controlled
  noise.  Reproducible from committed notebooks and test suite.
- **Exploratory** — infrastructure exists and initial results are available,
  but the experiment is not yet a complete comparative study.  Conclusions
  should be treated as methodological guidance rather than final quantitative
  proof.
- **Real-data validation** — not yet performed.  Requires calibrated images
  from known microscope instruments.

## Test suite

Run from the repository root:

```bash
rtk .venv/bin/python -m pytest               # fast suite: 163 passed, ~20 s
rtk .venv/bin/python -m pytest -m slow       # slow suite: 40 passed, ~7 min
rtk .venv/bin/python -m ruff check src/ tests/                       # lint gate
rtk .venv/bin/python examples/notebooks/check_docstring_params.py    # docstring gate
```

The suite is organised by what it actually guarantees against regression, from
strongest to weakest:

| Tier | What it checks | Representative tests |
|---|---|---|
| **Ground-truth recovery** | Fit a synthetic model of *known* parameters and assert the recovered vector matches the truth (`rtol` 1e-4) with near-zero residual (`rms < 1e-8`). Catches any error in the optics / estimation core. | `test_cmo_physical_model.py`, `test_cmo_telecentric_model.py`, `test_parallel_plate_physical_fit.py` |
| **Mathematical invariants** | Round-trip, channel symmetry, transverse gauge `O·d=0`, the `f_obj`/`telecentric_offset` degeneracy, Schur-complement shapes — at 1e-12. | `test_optical_ba_schur.py`, `test_inverse_problem_diagnostics.py`, `test_conventions.py`, `test_zernike_origin_field.py` |
| **Oracle equivalence / selection** | Each synthetic oracle recovers a near-zero residual; the model-selection pipeline picks the expected family (golden winner map). | `test_model_selection_oracles.py`, `test_physical_model_selection.py`, `test_rayfield_from_observations.py` |
| **API-shape equivalence (bit-exact)** | The multi-camera façade reproduces the validated stereo solver to `max abs diff == 0` on the left/right case. | `test_stereo_multicamera_equivalence.py` |
| **Anchored regression bounds** | End-to-end OpenCV / rayfield metrics bounded on *measured* values (raw 0.61 px, refined 0.39 px, skew / point-to-ray P95 ~1 mm) — a ~2× degradation trips the test, and `refined < raw` is enforced. | `test_calibration_regression_metrics.py` |
| **Paper-number guards** | See *Reproducibility* below. | `test_paper_numbers_regression.py`, `test_paper_reproduction_slow.py` |

Bounds are anchored on the current measured values with a margin, never set so
loosely that a real regression slips through (the earlier 5 px / 100 mm limits
that allowed a 10–1000× drift have been retightened).

## Reproducibility of the paper numbers

Two complementary tests pin the CMO manuscript's headline numbers and localise
any drift to a single layer:

- **`test_paper_numbers_regression.py`** (fast) — pins *paper ↔ tracked JSON*.
  It asserts the published numbers against the computation artefacts mapped in
  `docs/assets/cmo_paper/AUDIT.md` (Zernike 57p RMS 0.47 px; CMO 26p
  1.06 px / P50 0.87 / P95 1.84; corner-BA 0.88 px; baseline 24.9 mm;
  WD 64.7 mm; θ 22.6°). Only git-tracked assets are used, so it runs on a fresh
  clone.
- **`test_paper_reproduction_slow.py`** (slow) — pins *tracked input ↔ recompute*.
  It **recomputes** the 26p reprojection from the raw corner pixels, board poses
  and 26-parameter vector in `intermediate_state.npz` through the production
  `CMOTelecentricStereoModel`, closing the compute → paper loop. If the model
  evaluation or its parameterisation regresses, the headline 1.06 px is no
  longer reproducible and this fails.

Run together they pinpoint the fault layer: if the first passes but the second
fails, the paper and the JSON agree while the JSON is no longer reproducible
from the committed inputs. (This pair caught exactly such a break once — a
later principal-point constraint had silently diverged `intermediate_state.npz`
from the published 1.06 px; it was repaired by restoring the free-principal-point
fit the paper actually used, after a `git bisect` localised the offending
commit.)

To rebuild the underlying real-data artefacts from the raw Pycaso images:

```bash
rtk .venv/bin/python examples/notebooks/save_intermediate_state.py   # regenerate intermediate_state.npz
```

A self-contained Zenodo archive (concept DOI 10.5281/zenodo.20444215) bundles
the manuscript, figures, tables and data needed to rebuild the paper offline.
