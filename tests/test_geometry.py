import numpy as np

from stereocomplex.core.geometry import (
    PinholeCamera,
    pixel_to_sensor_um,
    sensor_um_to_pixel,
    triangulate_midpoint,
)
from stereocomplex.meta import parse_view_meta


def _view_meta(width: int = 640, height: int = 480, pitch_um: float = 3.45):
    return parse_view_meta(
        {
            "schema_version": "stereocomplex.meta.v0",
            "sensor": {"pixel_pitch_um": pitch_um, "binning_xy": [1, 1]},
            "preprocess": {"crop_xywh_px": [0, 0, width, height], "resize_xy": [1.0, 1.0]},
            "image": {"width_px": width, "height_px": height, "bit_depth": 8, "gamma": 1.0},
        }
    )


def test_pixel_sensor_roundtrip():
    view = _view_meta()
    rng = np.random.default_rng(0)
    u = rng.uniform(0, view.image.width_px - 1, size=(1000,))
    v = rng.uniform(0, view.image.height_px - 1, size=(1000,))
    x_um, y_um = pixel_to_sensor_um(view, u, v)
    u2, v2 = sensor_um_to_pixel(view, x_um, y_um)
    assert np.max(np.abs(u2 - u)) < 1e-9
    assert np.max(np.abs(v2 - v)) < 1e-9


def test_triangulation_midpoint_hits_known_point():
    target = np.array([10.0, -5.0, 500.0], dtype=np.float64)
    o1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    o2 = np.array([100.0, 0.0, 0.0], dtype=np.float64)
    d1 = target - o1
    d2 = target - o2
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)
    xyz, dist = triangulate_midpoint(o1, d1, o2, d2)
    assert np.linalg.norm(xyz - target) < 1e-6
    assert dist < 1e-9

