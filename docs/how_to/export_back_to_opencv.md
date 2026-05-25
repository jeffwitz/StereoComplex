# Export back to OpenCV

Once you have a `StereoOpenCVCalibrationResult` or
`StereoCentralRayFieldFitResult`, you can export it to native OpenCV
format:

```python
from stereocomplex.api import fit_opencv_stereo_from_dataset, assess_calibration

result = fit_opencv_stereo_from_dataset(...)

# Option A — OpenCV stereo calibration
K1, d1, K2, d2, R, T = result.to_opencv()

# Option B — StereoCentralRayField model (exports a sampled rayfield as pinhole matrices)
from stereocomplex.api.model_io import save_stereo_central_rayfield
save_stereo_central_rayfield(Path("my_model"), result.model)
```

The exported matrices are compatible with `cv2.stereoRectify`,
`cv2.triangulatePoints`, and any other OpenCV stereo pipeline.
