# External validation of the specimen relief — note for §4.8/§5.6

**Status:** the point-wise comparison is now SOLVED and VERSIONED as a figure
(see next section). The lower part of this note (native-Pycaso table, name-
collision and constraint-vs-flexibility narrative) is kept as background; its
numbers predate the fix and use the old amplitude-only ECC registration
(cc≈0.72) — they are NOT the versioned result.

## Versioned, reproducible result (use this)

Figure: `docs/assets/cmo_paper/figure_external_profilo_relief/` +
`examples/notebooks/generate_fig_profilo_relief_comparison.py`. Two fixes lifted
the comparison from cc≈0.72 to cc≈0.96, point-wise:

1. **Registration.** Rigid similarity (scale≈1.03), derived by *letter-mask IoU*
   on the embossed "E" (IoU 0.88) — flow-independent, replaces the opaque ECC
   homography. Per-window plane fit on the "E" footprint.
2. **DIS optical flow.** Sent the FULL image (not a 300 px-cropped ROI) and
   enabled variational refinement (`alpha=120, epsilon=1e-4, 50 iters`). Relief
   correlation against profilometry climbed 0.865 → 0.916 (VR) → 0.955 (full
   image) → 0.958.

| Surface | relief std | vs profilo | cc vs profilo |
|---|---|---|---|
| **Profilometry (ground truth)** | **20.8 µm** | **1.00×** | — |
| Soloff (deg 3) | 22.8 µm | 1.09× | 0.958 |
| **CMO 26p (this work)** | **23.0 µm** | **1.10×** | 0.958 |
| **Zernike rayfield (this work)** | **34.0 µm** | **1.63×** | 0.958 |

**Headline:** CMO 26p and Soloff are *indistinguishable* — `cc = 0.9997`,
RMS difference `0.57 µm`, an order of magnitude below their ~2 µm distance to
the profilometry. The compact 26-parameter physical model is as faithful as the
unconstrained polynomial; the per-pixel Zernike ray-field over-amplifies (1.63×).

---

## (Background — superseded) earlier draft, native-Pycaso cross-check

## The result

A profilometric scan of the same coin specimen provides an external 3-D ground
truth. We register it laterally to the dense reconstruction (letter "EN" region)
and compare the plane-normalised relief amplitude. We also run the **native
Pycaso** calibration toolbox (the reference Soloff implementation, opencv 4.5,
unpatched algorithm) on the same 11 calibration planes and apply it to the same
dense correspondences. Relief standard deviation on ~100k common points:

| Method | relief std | vs profilo | family |
|---|---|---|---|
| **Profilometry (ground truth)** | **19.3 µm** | **1.00×** | external |
| Pycaso Zernike-aberration (pform 3) | 20.8 µm | 1.08× | constrained global aberration basis |
| Pycaso Zernike-aberration (pform 6) | 21.6 µm | 1.12× | constrained |
| Pycaso Soloff (forward + LM inversion) | 21.7 µm | 1.13× | forward + numerical inversion |
| Pycaso Soloff–Zernike (Z-init + LM) | 21.7 µm | 1.13× | = Soloff (faster init) |
| **CMO 26p (this work)** | **23.2 µm** | **1.21×** | compact physical model |
| Direct 2D→3D regression, deg 3 (our "Soloff") | 23.5 µm | 1.22× | inverse-poly regression |
| **Zernike rayfield (this work)** | **34.1 µm** | **1.77×** | per-pixel non-central ray field |
| Pycaso direct 2D→3D, deg 4 | 53.2 µm | 2.76× | unstable high-degree inverse poly |

## Two readings, both externally anchored

1. **The CMO 26p is metrologically faithful.** Its relief (23 µm) sits with the
   profilometric truth (19 µm) and the independent native Soloff (22 µm); the
   modest ~20 % excess is consistent with dense-stereo (DIS) correspondence noise
   inflating the standard deviation (the profilometer is far cleaner).

2. **The Zernike rayfield over-amplifies the dense relief by ~1.8×.** This is not
   a property of the Zernike basis, nor of "non-parametric" calibration in
   general — it is specific to the **per-pixel, non-central ray-field
   parameterisation** (57–93 coefficients per channel), whose freedom converts
   local correspondence geometry/noise into amplified depth.

## Critical clarification — "Zernike rayfield" ≠ classical Zernike calibration

There is a **name collision** that must be defused in the paper. The classical
"Zernike" stereo-calibration method (e.g. Pycaso's `Zernike_*`) fits ~12 **global
optical-aberration coefficients** (tilt, defocus, astigmatism, coma, trefoil,
sphericity, …) in a forward projection model — it is **constrained and low-order**,
and here it is the **single most faithful** method tested (1.08×). Our "Zernike
rayfield" is the **opposite**: a high-dimensional **per-pixel 3-D line field**
(origin + direction, Zernike polynomials over the *image plane*), and it is the
one that **over-amplifies**. Same word, opposite behaviour. A reader familiar
with classical Zernike calibration will otherwise draw the exact wrong conclusion.

## The organising axis: constraint vs. flexibility

Every method lines up on one axis — *how constrained the model is*:

- **Faithful (1.08–1.22× truth):** Pycaso Zernike-aberration, Pycaso Soloff (LM),
  CMO 26p, direct deg-3 — all physically constrained or low-order, well-conditioned.
- **Over-amplifying:** our per-pixel Zernike rayfield (1.77×) and the deg-4 direct
  inverse polynomial (2.76×) — both over-flexible.

## Methodological honesty note (separate issue)

Our paper's "Soloff polynomial baseline" (§5.6) is, mathematically, Pycaso's
**`direct`** method (a 2D→3D inverse-polynomial regression), **not** the true
Soloff (forward 3D→2D + Levenberg–Marquardt inversion). The true Soloff is ~9 %
more faithful (21.7 vs 23.5 µm). Either rename our baseline "direct / inverse-
polynomial" or adopt the true Soloff (now available) as a stronger, correctly-
named baseline.

## To make this paper-grade

- [DONE] Tighten the lateral registration → rigid letter-mask IoU 0.88,
  point-wise cc 0.958 (was cc≈0.72 amplitude-only).
- [DONE] Improve the DIS flow (full image + variational refinement) and version
  the figure end-to-end (`figure_external_profilo_relief/`).
- [optional] Re-run the *native* Pycaso Soloff/Zernike-aberration densely with
  the improved flow and add them as extra panels — needs their dense outputs.
- Confirm the profilometer unit (µm) — corroborated by the versioned
  Soloff/CMO/profilo agreement (22.8 / 23.0 / 20.8 µm).
