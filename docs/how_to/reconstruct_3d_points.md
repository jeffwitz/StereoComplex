# Reconstruct 3‑D points

Once you have a calibrated stereo model, reconstruct 3‑D points from pixel
correspondences.

```python
from stereocomplex.metrics import reconstruct_points_central_stereo

# uv_left, uv_right are (N, 2) arrays of corresponding corners
rec = reconstruct_points_central_stereo(
    left_pixels=uv_left,
    right_pixels=uv_right,
    model=rayfield_result.model,
)

import numpy as np
print(f"Reconstructed {rec.points_3d_mm.shape[0]} points")
print(f"RMS ray gap: {np.mean(rec.ray_gap_mm):.4f} mm")
```

For non‑central models, use `reconstruct_points_with_origin_fields`
instead.  See the [Reference: public API](../reference/public_api) for
full signatures.
