# Mechanical-observable results — final audit

These results are generated from the committed held-out prediction arrays by `.github/workflows/strain-mechanical-observables.yml`. They are not hand-copied from the earlier manuscript.

## 1. Gauge length exposes the mechanical ranking

On the **fine 0.08 mm sweep**, the apparent-strain ranking separates progressively with gauge length.

Horizontal gauges:

| Gauge length | Ray field | Soloff-type | Direct polynomial |
|---:|---:|---:|---:|
| 0.3 mm | 765 µε | 894 µε | 753 µε |
| 0.6 mm | 391 µε | 462 µε | 458 µε |
| 1.2 mm | 194 µε | 264 µε | 345 µε |
| 1.8 mm | 136 µε | 213 µε | 321 µε |
| 2.4 mm | 108 µε | 187 µε | 311 µε |

Vertical gauges show the same behaviour: at 2.4 mm the values are approximately 99, 166 and 301 µε respectively. The direct mapping, despite giving the best nominal-depth result on the fine sweep, approaches a persistent ~300 µε false-strain floor. The ray field instead continues to decrease approximately as `1/L` over most of the available range.

The wide 0.70 mm sweep behaves differently: the models remain much closer over most gauge lengths. This is useful rather than inconvenient. Scale-dependent separation is not an algebraic property of ray representations; it is a property of the spatial error field produced by a calibration on a given acquisition.

## 2. The first-order spatial-increment relation is experimentally verified

For each held-out rigid translation the cloud is aligned to the reference with a **proper SE(3) fit only**: no reflection and no scale. The gauge strain predicted from the longitudinal increment of the non-rigid residual field,

\[
\delta\varepsilon_{ij,t}\simeq
\frac{\mathbf t_{ij,t}^{T}}{L_{ij}}
\left[\mathbf e_t(\mathbf x_j)-\mathbf e_t(\mathbf x_i)\right],
\]

agrees almost exactly with the exact finite gauge-length change. Representative fine-sweep correlations at 1.2 mm are `0.99998` for the ray field, `0.99937` for Soloff-type reconstruction and `1.00000` for the direct polynomial. The relation remains close to unity over the complete gauge-length sweep.

This converts the previous qualitative interpretation into a measured mechanism: the mechanically relevant strain error is the **spatial increment** of the reconstruction residual, not its pointwise amplitude.

## 3. A two-term scale law exposes the mechanism

The exact RMS curves are accurately described by the empirical diagnostic law

\[
\sigma_\varepsilon^2(L)=\left(\frac{A}{L}\right)^2+B^2,
\]

where `A` is a short-range endpoint-increment scale and `B` a coherent strain floor that survives increasing gauge length.

For the fine sweep:

| Model | A horizontal | B horizontal | A vertical | B vertical | fit R² |
|---|---:|---:|---:|---:|---:|
| Ray field | 0.229 µm | 55.8 µε | 0.207 µm | 49.6 µε | >0.9997 |
| Soloff-type | 0.265 µm | 140.7 µε | 0.261 µm | 150.9 µε | >0.998 |
| Direct polynomial | 0.207 µm | 300.6 µε | 0.198 µm | 288.0 µε | >0.9999 |

The short-range displacement scales `A` are all of the same order (~0.2–0.27 µm). The major difference therefore lies in the coherent strain floor `B`, not in a dramatically smaller local point error for the ray field. On this acquisition the direct mapping carries a coherent distortion corresponding to roughly 0.03% false strain, whereas the ray field has a ~5–6 times smaller coherent floor.

## 4. Spatial maps support the same interpretation

At a nominal 1.2 mm gauge, ray-field pseudo-strain is spatially mixed around zero, Soloff retains a visible coherent component, and the direct polynomial produces a broad same-sign strain field over much of the target. This explains why increasing gauge length averages down much of the ray-field contribution but not the direct-model contribution.

## 5. Compact CMO 26p: audited held-out wide-sweep result

The compact CMO model is evaluated only on the 91 reserved planes of the **wide acquisition**, because its ten-frame physical calibration was identified on that acquisition. It is not silently transferred to the fine series, whose image pose and scale differ.

At a nominal 1.2 mm gauge the CMO model gives **473.1 µε horizontally** and **356.9 µε vertically**. The fitted scale law gives:

| Direction | A | B | R² |
|---|---:|---:|---:|
| Horizontal | 0.491 µm | 207.1 µε | 0.99465 |
| Vertical | 0.355 µm | 173.4 µε | 0.99567 |

The median closest-ray gap over all raw wide-series correspondences is 21.7 µm. This is an internal triangulation diagnostic on raw held-out ChArUco corners, not a substitute for pixel calibration RMS or for the dense specimen-comparison gap.

The CMO result is deliberately not presented as a win. On the wide sweep it lies in the same mechanical-consistency band as the ray, Soloff and direct reconstructions and does not define the lowest apparent strain. This negative result strengthens the methodological claim: **physical interpretability and a low calibration residual are not substitutes for validation in the mechanical observable itself.**

## 6. Final manuscript claim

> Pointwise calibration accuracy does not determine strain accuracy. In a rigid-motion experiment, strain error is governed by the spatial increment structure of reconstruction errors over the measurement gauge. Models with comparable sub-micrometre short-range endpoint errors can exhibit coherent strain floors differing by a factor of about six. A compact physical CMO model is useful for interpretable optical reconstruction, but its mechanical performance must still be evaluated independently in strain space.

This is the mechanical centre of the final *Strain*-oriented manuscript.