# Fit a non‑central model from real images

Equivalent to the central rayfield path, but with Zernike origin fields.

```python
from stereocomplex.calibration import fit_stereo_zernike_origin_field

result = fit_stereo_zernike_origin_field(
    observations=dataset,
    K_left=K_left, K_right=K_right,
    T_right_left_initial=np.eye(4),
    max_order=4,
)
```

This gives you per‑pixel ray origins O(u,v) and directions d(u,v).
The diagnostic output includes `origin_rms_mm`, `direction_rms_rad`,
and per‑frame statistics.

For the full walkthrough, see the notebook
`examples/notebooks/05_noncentral_calibration_from_images.ipynb`.
