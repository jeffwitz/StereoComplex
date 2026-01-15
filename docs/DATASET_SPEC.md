# Spécification dataset (v0)

Structure :

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

## `meta.json` (par scène)

- `schema_version`: `"stereocomplex.dataset.v0"`
- `stereo.left` / `stereo.right` : métadonnées de vue (pitch obligatoire)
- `board` : dimensions physiques de la mire (mm)
- `sim_params` : paramètres de synthèse (optionnel, audit)

### `sim_params` (CPU/OptiX)

Champs possibles (selon générateur) :

- `camera_model` (ex: `"pinhole"`)
- `distortion_model` : `"none"` ou `"brown"`
- `distortion_left`, `distortion_right` : coefficients Brown `{k1,k2,p1,p2,k3}`
- `image_format` : `"png"` ou `"webp"`
- `outside_mask` : `"none"` ou `"hard"` (fond noir hors mire)

### `board.type = "charuco"`

Champs requis :

- `square_size_mm`, `marker_size_mm`
- `squares_x`, `squares_y`
- `aruco_dictionary` (ex: `"DICT_4X4_1000"`)
- optionnel : `pixels_per_square` (utile côté simulateur CPU)
- optionnel : `texture_interp` (ex: `"linear"`, `"cubic"`, `"lanczos4"`)

## `frames.jsonl`

Une ligne JSON par frame : noms de fichiers + poses/paramètres utiles (synthèse).

Notes :

- Les images peuvent être en `png` (par défaut) ou `webp` lossless, selon le générateur.

## `gt_points.npz`

Tableaux numpy :

- `frame_id` : `(N,)` indices frame
- `XYZ_world_mm` : `(N, 3)` points 3D sur le plan (mm)
- `uv_left_px` : `(N, 2)` projections (px)
- `uv_right_px` : `(N, 2)` projections (px)

## `gt_charuco_corners.npz` (si `board.type == "charuco"`)

Tableaux numpy :

- `frame_id` : `(N,)`
- `corner_id` : `(N,)` id stable (row-major dans la grille (rows-1, cols-1))
- `XYZ_world_mm` : `(N, 3)`
- `uv_left_px` : `(N, 2)`
- `uv_right_px` : `(N, 2)`
