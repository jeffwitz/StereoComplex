# Conventions

## Coordinate frames

- Image: `u` to the right, `v` down (pixel coordinates).
- Sensor: `x_um` to the right, `y_um` down, origin at the center of the **crop** (before resize).
- World (simulation): units in **mm**.

## Pixel centers

Coordinates `(u_px, v_px)` refer to pixel *centers*. Internally, we use `(u + 0.5, v + 0.5)`
to convert indices to continuous coordinates.

Practical note: some OpenCV APIs return corner coordinates in a convention that is effectively
shifted by 0.5 px. When comparing against dataset GT, the evaluation applies the appropriate
correction depending on the method (see `docs/CHARUCO_IDENTIFICATION.md`).
