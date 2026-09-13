# Mechanical-observable results — first pass

These results are generated on the committed held-out prediction arrays by `.github/workflows/strain-mechanical-observables.yml`. They are not hand-copied from the old manuscript.

## 1. The gauge-length result is substantially stronger than the original 1.2 mm comparison

On the **fine 0.08 mm sweep**, the apparent-strain ranking separates progressively with gauge length.

Horizontal gauges:

| Gauge length | Ray field | Soloff-type | Direct polynomial |
|---:|---:|---:|---:|
| 0.3 mm | 765 µε | 894 µε | 753 µε |
| 0.6 mm | 391 µε | 462 µε | 458 µε |
| 1.2 mm | 194 µε | 264 µε | 345 µε |
| 1.8 mm | 136 µε | 213 µε | 321 µε |
| 2.4 mm | 108 µε | 187 µε | 311 µε |

Vertical gauges show the same behaviour: at 2.4 mm the values are approximately 99, 166 and 301 µε respectively.

Thus the direct mapping, which is the best model for nominal depth on the fine sweep, does **not** converge toward the lowest strain error as the gauge is enlarged. Its apparent strain approaches a persistent ~300 µε floor. The ray field instead continues to decrease approximately as `1/L`.

The wide 0.70 mm sweep behaves differently: the three models are much closer over most gauge lengths. This is useful rather than inconvenient: the scale-dependent separation is not a tautological property of the ray representation; it is specific to the fine-motion experiment in which the original depth/strain ranking reversal was observed.

## 2. The first-order spatial-increment relation is experimentally verified

For each held-out rigid translation the cloud was first aligned to the reference using a **proper SE(3) fit only** (no reflection, no scale). The gauge strain predicted from the longitudinal increment of the non-rigid residual field,

\[
\delta\varepsilon_{ij,t}\simeq
\frac{\mathbf t_{ij,t}^{T}}{L_{ij}}
\left[\mathbf e_t(\mathbf x_j)-\mathbf e_t(\mathbf x_i)\right],
\]

agrees almost exactly with the exact finite gauge-length change.

Representative fine-sweep correlations at 1.2 mm are:

- ray field: `corr = 0.99998`;
- Soloff-type: `corr = 0.99937`;
- direct polynomial: `corr = 1.00000`.

The relation remains close to unity over the full gauge-length sweep. This turns the previous qualitative explanation into a measured mechanism: the relevant strain error is the **spatial increment** of the reconstruction residual, not its pointwise amplitude.

## 3. A two-term scale law exposes the mechanism

The exact RMS curves are accurately described by the empirical diagnostic law

\[
\sigma_\varepsilon^2(L)=\left(\frac{A}{L}\right)^2+B^2,
\]

where `A` is a short-range endpoint-increment scale and `B` is a coherent strain floor that survives increasing gauge length.

For the fine sweep:

| Model | A horizontal | B horizontal | A vertical | B vertical | fit R² |
|---|---:|---:|---:|---:|---:|
| Ray field | 0.229 µm | 55.8 µε | 0.207 µm | 49.6 µε | >0.9997 |
| Soloff-type | 0.265 µm | 140.7 µε | 0.261 µm | 150.9 µε | >0.998 |
| Direct polynomial | 0.207 µm | 300.6 µε | 0.198 µm | 288.0 µε | >0.9999 |

The short-range displacement scales `A` are all of the same order (~0.2–0.27 µm). The large difference therefore lies in the **coherent strain floor B**, not in a dramatically smaller local point error for the ray field.

This is a much stronger interpretation than the first draft's statement that the ranking merely reverses. On the fine sweep, the direct mapping has excellent point/depth behaviour but carries a spatially coherent distortion that behaves like a roughly 0.03% false strain. The ray field has a similar local endpoint-increment scale but a ~5–6 times smaller coherent strain floor.

## 4. The spatial maps support the same interpretation

The representative 1.2 mm pseudo-strain maps show:

- the ray-field errors are spatially mixed around zero;
- the Soloff maps contain a visible coherent component;
- the direct polynomial produces a broad same-sign strain field over most of the target.

This spatial pattern explains why increasing gauge length averages down the ray-field error but not the direct-model error.

## 5. Consequence for the manuscript claim

The main result can now be stated more strongly, while remaining supported by the data:

> Pointwise calibration accuracy does not determine strain accuracy. In a rigid-motion experiment, the strain error is governed by the spatial increment structure of reconstruction errors. On the fine Pycaso sweep, models with comparable sub-micrometre short-range endpoint error exhibit coherent strain floors differing by about a factor of six (~50–56 µε for the ray field versus ~288–301 µε for the direct mapping), even though the direct mapping is superior in nominal depth.

This result should become the mechanical centre of the *Strain* manuscript. The CMO model-identification contribution can then be compressed around it rather than carrying the full 40+ page development.

## Next quantitative step

The remaining important calculation is to place the compact **CMO 26p physical model** on the same held-out mechanical metrics. This should be done before claiming that the compact physical CMO model itself is superior for strain; at present the scale-dependent result compares the flexible ray field, Soloff-type and direct polynomial reconstructions.
