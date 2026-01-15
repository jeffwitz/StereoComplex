# Dataset specification (v0)

Layout:

```
dataset/v0/
  manifest.json
  train|val|test/
    scene_0000/
      meta.json
      frames.jsonl
      left/000000.png    (ou .webp)
      right/000000.png   (ou .webp)
      gt_points.npz
```

## `meta.json` (per scene)

- `schema_version`: `"stereocomplex.dataset.v0"`
- `stereo.left` / `stereo.right`: view metadata (pixel pitch is mandatory)
- `board`: target physical dimensions (mm)
- `sim_params`: simulation parameters (optional, auditing)

### `sim_params` (CPU/OptiX)

Possible fields (generator-dependent):

- `camera_model` (e.g. `"pinhole"`)
- `distortion_model`: `"none"` or `"brown"`
- `distortion_left`, `distortion_right`: Brown coefficients `{k1,k2,p1,p2,k3}`
- `image_format`: `"png"` or `"webp"`
- `outside_mask`: `"none"` or `"hard"` (black background outside the board)

### `board.type = "charuco"`

Required fields:

- `square_size_mm`, `marker_size_mm`
- `squares_x`, `squares_y`
- `aruco_dictionary` (ex: `"DICT_4X4_1000"`)
- optional: `pixels_per_square` (useful for the CPU simulator)
- optional: `texture_interp` (e.g. `"linear"`, `"cubic"`, `"lanczos4"`)

## `frames.jsonl`

One JSON object per frame: filenames + useful poses/parameters (simulation).

Notes :

- Images can be stored as `png` (default) or lossless `webp`, depending on the generator.

## `gt_points.npz`

NumPy arrays:

- `frame_id`: `(N,)` frame indices
- `XYZ_world_mm`: `(N, 3)` 3D points on the plane (mm)
- `uv_left_px`: `(N, 2)` projections (px)
- `uv_right_px`: `(N, 2)` projections (px)

## `gt_charuco_corners.npz` (if `board.type == "charuco"`)

NumPy arrays:

- `frame_id` : `(N,)`
- `corner_id`: `(N,)` stable id (row-major on the (rows-1, cols-1) corner grid)
- `XYZ_world_mm` : `(N, 3)`
- `uv_left_px` : `(N, 2)`
- `uv_right_px` : `(N, 2)`
