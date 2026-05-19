# StereoComplex — project summary (2026-05-20)

## Project purpose

StereoComplex is a Python library for non-central stereo camera calibration via
Zernike rayfields. Each pixel gets its own 3D ray, replacing the pinhole abstraction.
It identifies optical architectures through BIC-based model selection.

## CMO paper status

**v1.0-submission-ready.** 26 pages, 31 references, 11 figures, 5 tables.

Core: double TPS → Zernike rayfield BA → BIC model selection → CMO+SE(3) 26p at
1.06 px reprojection (corner BA: 1.06→0.88 px). Specimen sanity check with DIS flow,
Zernike vs CMO rigid removal (SE3/Sim3/anisotropic convergence diagnosis).
Bibliography cleaned: 31 verified entries, 0 undefined citations.

Remaining human tasks: deposit Zenodo archive, confirm contact email.

## Key files

| Path | Role |
|---|---|
| `src/stereocomplex/` | Library source |
| `tests/` | 109 tests pass |
| `examples/notebooks/` | Notebooks 00-11 |
| `paper/cmo/manuscript.tex` | 26-page manuscript |
| `paper/cmo/references.bib` | 31 verified entries |
| `paper/cmo/figures/` | 11 PNG figures |
| `paper/cmo/tables/` | 5 LaTeX tables |
| `docs/assets/pycaso_real_data/` | 45+ artifacts |

## Active conventions

- deep-claude: fresh session + CLAUDE.md only (no --resume c7c56802, too large)
- Flags: --pro --dangerously-skip-permissions --allowedTools "Read,Write,Edit,Bash"
- Figures: matplotlib classic, serif fonts, dpi=150
- All artefacts saved as JSON
- Corner BA: refine_26p_corners_fast.py (foreground, 10 min)
- User audits CDC via ChatGPT before execution
- Watcher: persistent wait_telegram.py with watch_patterns TELEGRAM_MSG#
