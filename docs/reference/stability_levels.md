# Reference: stability levels

API stability commitments as of v0.7.0-alphaari.

## Stable

No breaking changes in patch or minor releases.

- OpenCV stereo path (`fit_opencv_stereo_*`)
- Ray2D corner refinement (`predict_charuco_points`)
- Central stereo rayfield (`fit_stereo_central_rayfield_*`)
- Non‑central Zernike origin field (`fit_stereo_zernike_origin_field`)
- Model import/export (`save/load_stereo_central_rayfield`)
- Reconstruction diagnostics (`reconstruct_points_*`)

## Advanced

Stable interface, expert use.  May change after deprecation notice.

- Physical model selection (`fit_physical_model_to_rayfield`)
- CMO models (`fit_cmo_*_to_rayfields`)
- Schur diagnostics (`optical_ba`)
- Paper reproduction helpers (`benchmarks/`, `synthetic/`)

## Experimental

No stability promise.  Signatures, layouts, and return types will change
without notice.

- N‑camera facades (`calibrate(cameras=...)`)
- `MultiCameraCharucoObservationSet` and containers
- `CMOTelecentricNModel`
- `fit_zernike_rayfields_from_multi_camera_observations`

See also: [RELEASE_READINESS](../RELEASE_READINESS) for the full snapshot
definition.
