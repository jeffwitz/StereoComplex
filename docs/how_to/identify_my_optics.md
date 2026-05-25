# Identify which optical model fits my system

Once you have a Zernike rayfield, select the best physical model
(pinhole, Brown-Conrady, parallel-plate, CMO telecentric, etc.) by BIC.

```python
from stereocomplex.physics.model_selection import (
    fit_physical_model_to_rayfield,
    OpticalModelCandidate,
)

candidates = [
    OpticalModelCandidate("pinhole", CentralPinholeModel, ...),
    OpticalModelCandidate("brown", CentralBrownConradyModel, ...),
    OpticalModelCandidate("plate", PinholeParallelPlateModel, ...),
    OpticalModelCandidate("cmo_tc", CMOTelecentricStereoModel, ...),
]

for c in candidates:
    fit = fit_physical_model_to_rayfield(
        c.model_class, target_field=left_field,
        K=K_left, image_size=(640, 480),
        initial_parameters=c.initial_guess(),
    )
    print(f"{c.name}: rms={fit.rms_mm:.4f} mm, n_params={fit.model.n_parameters}")
```

For the full theory, see [Explanation: Ray‑space BIC](../explanation/ray_space_bic).
