from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stereocomplex.meta import ViewMeta


def pixel_to_sensor_um(
    view: ViewMeta, u_px: np.ndarray, v_px: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map delivered image pixel coordinates (u,v) -> sensor-plane coordinates in µm.

    Convention: origin at center of crop (before resize), x right, y down.
    """
    u_px = np.asarray(u_px, dtype=np.float64)
    v_px = np.asarray(v_px, dtype=np.float64)

    _crop_x, _crop_y, crop_w, crop_h = view.preprocess.crop_xywh_px
    resize_x, resize_y = view.preprocess.resize_xy

    # Back-map to cropped (binned) sensor pixel coordinates (continuous, pixel centers).
    u_crop = (u_px + 0.5) / resize_x - 0.5
    v_crop = (v_px + 0.5) / resize_y - 0.5

    # Center crop-space around its middle.
    u0 = (crop_w - 1) / 2.0
    v0 = (crop_h - 1) / 2.0

    pitch_x_um = view.sensor.pixel_pitch_um * view.sensor.binning_xy[0]
    pitch_y_um = view.sensor.pixel_pitch_um * view.sensor.binning_xy[1]

    x_um = (u_crop - u0) * pitch_x_um
    y_um = (v_crop - v0) * pitch_y_um
    return x_um, y_um


def sensor_um_to_pixel(
    view: ViewMeta, x_um: np.ndarray, y_um: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse of `pixel_to_sensor_um` for the same conventions (origin at crop center).
    Returns delivered image pixel coordinates (u_px, v_px).
    """
    x_um = np.asarray(x_um, dtype=np.float64)
    y_um = np.asarray(y_um, dtype=np.float64)

    _crop_x, _crop_y, crop_w, crop_h = view.preprocess.crop_xywh_px
    resize_x, resize_y = view.preprocess.resize_xy

    pitch_x_um = view.sensor.pixel_pitch_um * view.sensor.binning_xy[0]
    pitch_y_um = view.sensor.pixel_pitch_um * view.sensor.binning_xy[1]

    u0 = (crop_w - 1) / 2.0
    v0 = (crop_h - 1) / 2.0
    u_crop = x_um / pitch_x_um + u0
    v_crop = y_um / pitch_y_um + v0

    u_px = (u_crop + 0.5) * resize_x - 0.5
    v_px = (v_crop + 0.5) * resize_y - 0.5
    return u_px, v_px


def pixel_grid_um(view: ViewMeta) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: returns (x_um, y_um) grids shaped (H,W) for the delivered image."""
    w, h = view.image.width_px, view.image.height_px
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    return pixel_to_sensor_um(view, uu, vv)


@dataclass(frozen=True)
class PinholeCamera:
    f_um: float

    def ray_directions_cam(self, x_um: np.ndarray, y_um: np.ndarray) -> np.ndarray:
        """Central ray directions in camera frame for a pixel grid."""
        x_mm = np.asarray(x_um, dtype=np.float64) / 1000.0
        y_mm = np.asarray(y_um, dtype=np.float64) / 1000.0
        f_mm = float(self.f_um) / 1000.0
        dirs = np.stack([x_mm, y_mm, np.full_like(x_mm, f_mm)], axis=-1)
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        return dirs / norms

    def ray_directions_cam_from_norm(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Rays from normalized camera coordinates x=X/Z, y=Y/Z.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        return dirs / norms


def triangulate_midpoint(
    o1_mm: np.ndarray, d1: np.ndarray, o2_mm: np.ndarray, d2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mid-point triangulation of two skew 3-D rays.

    Finds the pair of points (one on each ray) that minimise the Euclidean
    distance between the two rays, and returns their midpoint as the
    triangulated 3-D position.  The ray distance (gap) is the minimum
    distance between the two lines — zero when the rays intersect exactly.

    For parallel or nearly-parallel rays (denominator < 1e-12), the midpoint
    of the two ray origins is returned and the distance is the orthogonal
    distance from o1 to the line through o2.

    Parameters
    ----------
    o1_mm : ndarray, shape (N, 3) or (3,)
        Ray origins of the first set of rays, in millimetres.
    d1 : ndarray, shape (N, 3) or (3,)
        Ray directions for the first set (unit vectors, will be normalised).
    o2_mm : ndarray, shape (N, 3) or (3,)
        Ray origins of the second set of rays, in millimetres.
    d2 : ndarray, shape (N, 3) or (3,)
        Ray directions for the second set (unit vectors, will be normalised).

    Returns
    -------
    XYZ_mm : ndarray, shape (N, 3)
        Triangulated 3-D points (midpoints of closest-approach segments),
        in millimetres.
    ray_distance_mm : ndarray, shape (N,)
        Minimum distance between each ray pair, in millimetres.

    Notes
    -----
    This is a vectorised version working on stacks of ray pairs.  The
    gauge convention is the one implicit in the input origins — no
    transverse projection is applied here; the origins are used as given.
    """
    o1_mm = np.asarray(o1_mm, dtype=np.float64)
    o2_mm = np.asarray(o2_mm, dtype=np.float64)
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)

    # Solve for closest points on skew lines.
    w0 = o1_mm - o2_mm
    a = np.sum(d1 * d1, axis=-1)
    b = np.sum(d1 * d2, axis=-1)
    c = np.sum(d2 * d2, axis=-1)
    d = np.sum(d1 * w0, axis=-1)
    e = np.sum(d2 * w0, axis=-1)

    denom = a * c - b * b
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)

    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom

    p1 = o1_mm + t1[..., None] * d1
    p2 = o2_mm + t2[..., None] * d2
    xyz = 0.5 * (p1 + p2)
    dist = np.linalg.norm(p1 - p2, axis=-1)
    return xyz, dist
