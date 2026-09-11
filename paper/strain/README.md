# Depth and apparent-strain assessment

This directory contains a short manuscript draft aimed at the experimental
mechanics/DIC community, together with executed, reproducible analyses. It is a
new, deliberately narrower article; `paper/cmo` is unchanged.

**Main finding:** on the fine archived sweep, the direct affine reconstruction
best follows nominal depth, while the finite-plane ray field best preserves
1.2 mm virtual-gauge lengths during nominal rigid translation. This is not an
independent demonstration of better DIC strain accuracy.

## Files

- `manuscript.pdf`, `manuscript.tex`, `references.bib`: English article draft.
- `EDITORIAL_NOTE_FR.md`: originality, journal positioning, retained claims and limitations.
- `experiments/`: detection, model implementations, evaluation, simulations,
  independent numerical checks, spatial holdout, timing and figures.
- `results/`: observations, predictions, numerical results, per-file SHA-256
  hashes and verification report. The JSON files contain per-plane outputs,
  cross-validation folds and all 488 simulation runs.
- `figures/`: three manuscript figures, each in vector PDF and PNG.
- `table_main.tex`, `table_additional.tex`: automatically generated numerical tables.

## Provenance and dependencies

Code base: StereoComplex `ed422d6581d841771fbeeb7c0f596e5d05875766`.
Raw images: [PYCASO](https://github.com/LaboratoireMecaniqueLille/Pycaso)
`44fbf1aea2479ae4cf21ed0902a91867540dd7c3`.
Four source folders are used, all under `Exemple/Images_example`:
`left_calibration`, `right_calibration`, `left_calibration2`, `right_calibration2`.
Images remain in their original repository. No new physical measurement is used.

The executed environment uses Python 3.12.14, NumPy 2.3.5 and
`opencv-contrib-python-headless==4.13.0.92` (OpenCV reports 4.13.0).
SciPy, Matplotlib and Pillow are also needed; detailed runtime versions are
recorded in `results/environment.json`. Use one BLAS/OpenMP thread for comparable
small-matrix timings. A TeX Live installation with latexmk, natbib, lmodern,
microtype and caption compiles the manuscript.

## Reproduce from the repository root

Install the Python dependencies if necessary:

```bash
python -m pip install -r paper/strain/requirements.txt
```

Place the pinned PYCASO checkout next to StereoComplex. For example:

```bash
git clone https://github.com/LaboratoireMecaniqueLille/Pycaso.git ../Pycaso
git -C ../Pycaso checkout 44fbf1aea2479ae4cf21ed0902a91867540dd7c3
```

Execute the pipeline:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python paper/strain/experiments/detect.py --pycaso ../Pycaso
python paper/strain/experiments/evaluate.py
python paper/strain/experiments/synthetic.py
python paper/strain/experiments/verify.py
python paper/strain/experiments/additional.py
python paper/strain/experiments/figures.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/strain/manuscript.tex
```

The detection step can be skipped to reproduce downstream computations from the
included NPZ observations. Figure 1 still needs the two reference images from
PYCASO. The figure script expects the sibling-directory layout above.

## Exact protocol

The wide sweep uses 101 paired depths from 2.65 to 3.35 mm. Calibration indices
are `[0,10,20,30,40,50,60,80,90,100]`. The fine sweep uses 101 paired depths from
2.96 to 3.04 mm, with calibration indices `[0,11,22,33,44,56,67,78,89,100]`.
All other planes are reserved. Only corners observed in both images are used.
There is no missing-corner completion, TPS denoising or model-specific rejection.

OpenCV ChArUco parameters: 16 x 12 squares, 0.3 mm square side, 0.15 mm marker
side, DICT_6X6_250, legacy pattern, `checkMarkers=False`. ArUco parameters:
adaptive threshold windows 3 to 75 with step 4; minimum/maximum marker perimeter
rate 0.005/0.20; polygon approximation 0.03; corner distance 0.02; border distance
1 pixel; error correction 0.6; subpixel marker refinement. ChArUco corners are
then refined with `cornerSubPix`, half-window 5 x 5 pixels, 40 iterations,
epsilon 1e-4. Remaining parameters retain the specified OpenCV version defaults.

Orders are selected only within the ten calibration planes by leave-one-plane-out
depth RMSE, equally weighting each plane. Candidate orders: ray 1--4, direct 1--4,
forward complete degrees 1--4 plus Soloff 332. The latter is all total-degree-three
monomials except pure z cubed. Coordinates and design columns are normalised
using training observations. SVD uses `rcond=1e-12`, without ridge regularisation.
No board-pose optimisation is performed. This is not a benchmark of the full
upstream PYCASO package, whose calibration includes additional options.

The reference is frame 50 in each series. Its self-comparison is excluded from
the fine motion/gauge aggregate. Gauges span four squares horizontally or
vertically. The strain aggregate gives equal weight to each evaluated frame.
The pointwise depth aggregate gives equal weight to each retained point. These
are intentionally different observables. All full per-frame results are saved.

The additional spatial control fits even checkerboard parity corners on training
planes and evaluates odd parity corners on reserved planes, using the primary
selected orders. It tests spatial interpolation, with no retuning.

Synthetic controls use independent distorted-perspective projection, two depth
extents, four coordinate-noise levels and 20 paired seeds for each nonzero noise
level. Four fixed cubic representations give `2 * (1 + 3 * 20) * 4 = 488` fits.
Noiseless controls have one run. Three reserved planes are also stretched by
1000 microstrain in X, with zero Y extension. These are coordinate simulations,
not rendered speckle images and not a complete DIC simulation.

## What the results establish

The reported real-data scores are discrepancies from nominal coordinates and
rigid-target gauge consistency. Stage accuracy and target geometry are not
independently certified here. The two archived series are fitted separately;
they are not assumed to be the same unchanged optical configuration. Gauge
errors on fiducials at 1.2 mm cannot be converted directly into a full-field DIC
strain uncertainty at another spatial resolution.

The source PYCASO article reports a coin/profilometry comparison. The inspected
repositories contain coin images, but no numerical profilometry reference was
found. The published comparison is therefore not repackaged as a newly executed
independent validation. Physical optical-parameter identification, BIC rankings
from the long manuscript and Schur regularisation are outside this article.

Runtime results concern the provided vectorised Python implementations in a
shared compute environment. They are not an algorithm-independent performance
claim. Raw results are retained at full precision; manuscript tables round them.

The analysis is exploratory and retrospectively defined, not preregistered.
Authorship, editorial declarations and interpretation remain subject to author
review before submission; the delivered document is a manuscript draft.
