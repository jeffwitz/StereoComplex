# Reference: file formats

Model import/export schemas via `api/model_io.py`.

## `save_stereo_central_rayfield(dir)`

Writes two files:

| File | Format | Contents |
|---|---|---|
| `model.json` | JSON | `schema_version`, image size, disk parameters, per‑frame rvecs/tvecs, rig transform |
| `coeffs_left_x.npy` | NPY | Zernike coefficients for left camera, shape `(n_modes,)` |
| `coeffs_left_y.npy` | NPY | Zernike coefficients for left camera, shape `(n_modes,)` |
| `coeffs_right_x.npy` | NPY | Zernike coefficients for right camera |
| `coeffs_right_y.npy` | NPY | Zernike coefficients for right camera |

## `load_stereo_central_rayfield(dir)`

Reads the five files above.  Schema version must be
`"stereocomplex.model.stereo_central_rayfield.v0"`.
