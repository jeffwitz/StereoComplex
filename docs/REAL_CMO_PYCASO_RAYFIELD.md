# Real CMO microscope data: from ChArUco images to CMO-consistent rayfield descriptors

> **First real-data validation of the StereoComplex non-central pipeline.**

## What this page shows

StereoComplex is applied to **real CMO stereo microscope calibration images**
from the [Pycaso](https://github.com/LaboratoireMecaniqueLille/Pycaso)
open-source project.  The goal is not only to reduce reprojection residuals,
but to **measure a 3D rayfield and read physically interpretable CMO-consistent
descriptors from it** — without fitting a physical model.

## Pipeline

```text
ChArUco legacy detection (DICT_6X6_250, setLegacyPattern)
       ↓
Hessian corner completion (|det H| + Otsu + barycentre)
       ↓
Ray2D TPS denoising (predict_points_rayfield_tps_robust)
       ↓
Constrained Zernike rayfield O(0)+d(2), shared R+XY, per-pose Z
       ↓
CMO-consistent geometric descriptors read from (O, d)
       ↓
Simplified CMO model mismatch diagnostic (telecentricity)
```

## Dataset

| Property | Value |
|---|---|
| Microscope | CMO stereo (Pycaso example) |
| Sensor | 2048 × 2048 px |
| Board | ChArUco 16 × 12 squares, 0.3 mm |
| Dictionary | DICT_6X6_250 (legacy pattern) |
| Frames | 10 stereo pairs |
| Z range | 2.65 – 3.35 mm (Δ = 0.70 mm) |

## Main result

A standard central OpenCV stereo calibration **does not converge** to a
usable model under the tested configuration.  By contrast, the constrained
Zernike rayfield reaches **subpixel local pixel-equivalent residuals**
(0.47 px RMS baseline, 0.41 px with O(2)+d(3)).

From the measured Zernike rayfield at the centre pixel — *without any
numerical optimisation* — we read CMO-consistent geometric descriptors:

| Descriptor | Symbol | Value | Source |
|---|---|---|---|
| Stereo baseline | $b$ | 24.9 mm | $\|O_R - O_L\|$ |
| Working distance | $WD$ | 64.7 mm | Mean board Z from poses |
| Objective focal length | $f_{\text{obj}}$ | 62.2 mm | $WD - z_{\text{pupil}}$ |
| Sub-pupil depth | $z_{\text{pupil}}$ | 2.5 mm | $(|O_{L,z}|+|O_{R,z}|)/2$ |
| Convergence angle | $\theta$ | 22.6° | $\arccos(d_L \cdot d_R)$ |

## Why this matters

The measured rayfield is not just a flexible fit.  It exposes **physically
interpretable structure**: effective sub-pupil positions, chief-ray
convergence, and a quantifiable mismatch with a minimal perspective CMO model.

Comparing the Zernike $d_y$ (nearly constant, $\approx 0.059 \pm 0.04$) against
the perspective CMO model (linear gradient, range $3\times$ larger) reveals
that the real optics are **more object-space telecentric** than a simple
perspective CMO model predicts.  This mismatch is a *diagnostic*, not a
failure — it characterises the real instrument.

## Important caveat

These are **CMO-consistent geometric descriptors** read from a Zernike
rayfield under the transverse gauge $O(u,v)\cdot d(u,v)=0$.  They are
*not* fitted CMO physical parameters.  The comparison model uses fixed
principal point (image centre), zero distortion, fixed pixel pitch
(5.5 µm), and a guessed tube-lens focal length (50 mm).  These choices
contribute to the observed CMO mismatch and should be refined in a
dedicated physical model fit.

## Zernike order sweep

| Model | Params | RMS (px) | P95 (px) | Notes |
|---|---|---|---|---|
| O(0)+d(2) | 57 | 0.470 | 0.852 | baseline, most physical |
| O(1)+d(2) | 69 | 0.444 | 0.807 | |
| O(0)+d(3) | 81 | 0.419 | 0.804 | |
| O(1)+d(3) | 93 | 0.412 | 0.791 | |
| O(2)+d(3) | 111 | **0.409** | **0.786** | best, diminishing returns |
| O(1)+d(4) | 123 | 0.412 | 0.790 | |
| O(2)+d(4) | 141 | 0.410 | 0.789 | no further gain |

Physical parameter stability across orders:

- $WD$ is rock-solid (spread < 0.5 mm);
- $f_{\text{obj}}$ varies by ~1.5 mm (2 %);
- $b$ is sensitive to the origin-field order (20–25 mm), because higher O-orders
  can absorb spatial baseline variations into per-pixel origin structure.

**Recommendation:** O(0)+d(2) is the most physically interpretable model
(rigid sub-pupil per channel).  Higher orders reduce RMS modestly (−13 %)
but at the cost of mixing baseline into the origin gauge.

## Reproduce

```bash
PYTHONPATH=src python examples/notebooks/09_pycaso_real_data.py
```

Requires a local Pycaso clone at `examples/pycaso_data`:

```bash
git clone https://github.com/LaboratoireMecaniqueLille/Pycaso examples/pycaso_data
```

## See also

- :doc:`IDENTIFY_MY_OPTICS` — how to read physical descriptors from a rayfield
- :doc:`CMO_PHYSICAL_MODEL` — the shared-rig CMO model definition
- :doc:`DIRECT_VS_RAYFIELD_INVERSION` — why measure a rayfield before fitting optics
- :doc:`NOTEBOOKS` — all walkthrough notebooks
- [Notebook 09](../examples/notebooks/09_pycaso_real_data.py) — full source
