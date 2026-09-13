# Reframing the CMO study for *Strain*

## Editorial objective

This branch does **not** turn the existing 9-page `paper/strain` draft into a shorter note. The intended manuscript is a dense ~15-page *Strain* article that compresses the 40+ page CMO study while adding the missing mechanical validation layer.

The central claim is deliberately stronger than the first strain draft:

> A stereo calibration intended for deformation measurement must be validated through the spatial increments of reconstruction error at the mechanical gauge length and direction. Pointwise reprojection/depth accuracy is not sufficient. A measured non-central rayfield can be used to identify a compact physical CMO model, and rigid-target sequences then provide a direct experimental zero-strain consistency test of the resulting calibration.

The Pycaso CMO microscope is therefore not merely a difficult camera-calibration example. It is the experimental system on which a complete chain is demonstrated:

```
measure rays -> identify compact CMO physics -> reconstruct held-out rigid motions
-> quantify scale/direction-dependent pseudo-strain -> check independent specimen relief
```

## What is retained from the long CMO paper

Only the pieces needed to establish the physical calibration and its credibility should remain in the main Strain article:

1. Why a CMO microscope is non-central and why standard pinhole calibration fails.
2. The two-stage decomposition: flexible Zernike rayfield first, compact physical model second.
3. Residual-guided identification of the quasi-telecentric CMO + per-arm SE(3) family.
4. The final compact 26-parameter CMO result and a concise comparison with OpenCV, Soloff and the flexible rayfield.
5. The observability issue only to the extent needed to explain why the rayfield-mediated identification is preferable to blind joint optics/pose fitting. One compact Schur-spectrum figure or one paragraph is enough in the main text; the detailed regularisation study belongs in supplementary material.
6. The dense coin reconstruction and external profilometry, because they supply an independent axial check. The Y-axis reflection is a documented CMO coordinate convention and must be removed before SE(3) comparison; it is not a shape discrepancy.

## What moves out of the main article

The following CMO-paper material is useful for reproducibility but dilutes the *Strain* narrative and should move to supplementary material/repository documentation:

- the full Zernike order sweep and detailed BIC discussion;
- the complete progression through every rejected optical family;
- all alternative residual hypotheses;
- detailed SE(3) rotation/translation ablations;
- isotropic-vs-Schur prior sweeps and regularisation Pareto plots;
- sensitivity to the fixed reference focal length;
- number-of-frames sensitivity;
- implementation-level optimisation details;
- large tables that duplicate one another.

The main text should retain one model-identification figure that makes the logic visible: wrong central/perspective hypothesis -> telecentric CMO -> residual piston -> per-arm SE(3) -> usable compact model.

## New mechanical contribution

The previous short draft used only one virtual-gauge length (1.2 mm) and collapsed most of the available spatial information into one RMS. That is insufficiently strong. The new analysis uses the existing held-out reconstruction arrays to compute the following without refitting the calibration models.

### 1. Rigid-residual field

For reference cloud $\widehat{\mathbf X}_0$ and reconstructed cloud $\widehat{\mathbf X}_t$, fit only a proper rigid transform $(\mathbf R_t,\mathbf T_t)\in SE(3)$:

\[
\mathbf e_t(\mathbf x)=\widehat{\mathbf X}_t(\mathbf x)-
\left[\mathbf R_t\widehat{\mathbf X}_0(\mathbf x)+\mathbf T_t\right].
\]

No reflection and no scale are allowed. The residual is therefore the non-rigid deformation invented by the reconstruction chain during a physically rigid motion.

### 2. Exact virtual-gauge strain

For endpoints $i,j$,

\[
\varepsilon^{\rm app}_{ij,t}=
\frac{\|\widehat{\mathbf X}_{j,t}-\widehat{\mathbf X}_{i,t}\|}
     {\|\widehat{\mathbf X}_{j,0}-\widehat{\mathbf X}_{i,0}\|}-1.
\]

It is evaluated over all admissible horizontal, vertical and diagonal separations, not only the historical 1.2 mm gauges.

### 3. First-order prediction from spatial error increments

With the rigidly transported gauge tangent $\mathbf t_{ij,t}$ and reference length $L_{ij}$,

\[
\delta\varepsilon_{ij,t}\simeq
\frac{\mathbf t_{ij,t}^{T}}{L_{ij}}
\left[\mathbf e_t(\mathbf x_j)-\mathbf e_t(\mathbf x_i)\right].
\]

This equation is the mechanical core of the article. It predicts strain error from the spatial structure of the reconstruction residual and explains why a model can have a larger pointwise depth error yet a smaller strain error.

### 4. Longitudinal structure amplitude

Define

\[
D_{\parallel}(L,\theta)=
\left\langle
\left(\mathbf t^T[\mathbf e(\mathbf x+L\mathbf t)-\mathbf e(\mathbf x)]\right)^2
\right\rangle,
\]

so that, to first order,

\[
\sigma_{\varepsilon}(L,\theta)\simeq
\frac{\sqrt{D_{\parallel}(L,\theta)}}{L}.
\]

This converts the present dataset from a single 1.2-mm score into a scale- and direction-dependent strain-consistency characterisation.

## Target main-text figures

1. **Instrument + protocol.** CMO optical sketch, calibration target, wide/fine translations, training/test split.
2. **Compressed identification chain.** Measured rayfield -> candidate CMO -> residual structure -> CMO+SE(3), with the key RMS values only.
3. **Calibration/reconstruction comparison.** Compact table/plot: OpenCV, Zernike, CMO 26p, Soloff/direct as appropriate; distinguish training fit from held-out mechanical observables.
4. **Held-out depth and motion.** Depth and rigid-motion discrepancies over the two sweeps.
5. **Pseudo-strain maps.** Same rigid translation, common colour scale, model columns; horizontal and vertical 1.2-mm gauges.
6. **Gauge-length dependence.** $E_\varepsilon(L)$ for each model, wide and fine sweeps.
7. **Directional dependence.** $E_\varepsilon(L,\theta)$, ideally polar/heat-map representation.
8. **Structure-function prediction.** Exact gauge strain versus first-order prediction from rigid-residual increments; include correlation/error.
9. **Independent specimen check.** Reflection/convention removal -> SE(3) alignment -> depth-only scale difference -> profilometry comparison. This remains a 3D relief check, not a strain-ground-truth experiment.

Some panels can be combined to keep the final manuscript near 15 pages.

## Target conclusions (conditional on the new calculations)

These claims should be stated only if the new analysis supports them numerically:

1. Reprojection and pointwise 3D errors are insufficient metrics for selecting a stereo calibration for deformation measurement.
2. The relevant calibration error for strain is the spatial increment of reconstruction error over the mechanical gauge length.
3. Calibration performance is therefore intrinsically scale- and direction-dependent; a model ranking at one gauge length is not automatically transferable to another.
4. Rigid-target sequences provide an experimental zero-strain consistency test without requiring the rigid-body translation amplitude itself to be independently calibrated.
5. The compact CMO model should be described as mechanically preferable only if its held-out scale/direction metrics support that statement; the article must not infer this from reprojection error alone.

## Planned manuscript structure (~14--16 pages)

1. Introduction (1.5--2 p.)
2. CMO calibration in ray space: compressed physical identification (2.5--3 p.)
3. Mechanical validation framework: rigid residual, gauge strain, structure function (2.5--3 p.)
4. Experimental and synthetic protocols (1.5--2 p.)
5. Results (4--5 p.)
6. Discussion and limitations (2 p.)
7. Conclusion (0.5 p.)

## Current implementation state

`experiments/mechanical_observables.py` performs the gauge-length/direction and rigid-residual analysis directly from the saved prediction arrays. It writes a machine-readable JSON summary and maps, and generates the first four new mechanical figures. A GitHub Actions workflow executes this analysis on the branch so that the numerical conclusions are derived from committed data rather than copied by hand.
