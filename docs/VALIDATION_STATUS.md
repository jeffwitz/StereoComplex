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

## Test status

- **Fast tests:** 76 passed, 0 warnings, ~28 s (`pytest -m "not slow"`)
- **Full suite:** 113 passed, 0 warnings, ~9 min (`pytest -m ""`)

## Reproducibility

```bash
# Fast check (~60 s)
python examples/reproduce_docs_results.py --fast

# Full sweep (~15 min)
python examples/reproduce_docs_results.py --full
```
