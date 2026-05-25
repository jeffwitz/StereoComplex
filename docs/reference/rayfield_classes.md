# Reference: rayfield classes

The rayfield class hierarchy.

## `Ray2DField`

`core/rayfield2d.py` — 2‑D planar rayfield (homography + smooth correction).
Used for corner refinement; not a 3‑D camera model.

## `CentralRayFieldZernike`

`core/model_compact/central_rayfield.py` — central 3‑D rayfield: all rays
share one camera centre, but each pixel has its own Zernike‑modelled
direction. `ray(u,v) → (origin=0, direction)`.

## `ZernikeRayField`

`rayfields/zernike_origin_field.py` — full non‑central rayfield: per‑pixel
origin O(u,v) and direction d(u,v), both modelled by Zernike polynomials
in pupil coordinates.

## `MultiCameraZernikeRayField`

Container holding one `ZernikeRayField` per named camera channel.

| Method | Returns | Description |
|---|---|---|
| `ray(channel, u, v)` | `(O, d)` | Ray for a pixel in a named channel |
| `from_fields(fields)` | (classmethod) | Build from a dict of `{name: ZernikeRayField}` |
| `channel_names` | `tuple[str]` | Names in insertion order |
