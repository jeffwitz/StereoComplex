import pytest

from stereocomplex.meta import MetaValidationError, parse_view_meta


def test_parse_view_meta_ok():
    m = parse_view_meta(
        {
            "schema_version": "stereocomplex.meta.v0",
            "sensor": {"pixel_pitch_um": 3.45, "binning_xy": [2, 1]},
            "preprocess": {"crop_xywh_px": [10, 20, 320, 240], "resize_xy": [2.0, 2.0]},
            "image": {"width_px": 640, "height_px": 480, "bit_depth": 8, "gamma": 1.0},
        }
    )
    assert m.sensor.binning_xy == (2, 1)
    assert m.image.width_px == 640


def test_parse_view_meta_rejects_missing_pitch():
    with pytest.raises(MetaValidationError):
        parse_view_meta(
            {
                "schema_version": "stereocomplex.meta.v0",
                "sensor": {"binning_xy": [1, 1]},
                "preprocess": {"crop_xywh_px": [0, 0, 10, 10], "resize_xy": [1.0, 1.0]},
                "image": {"width_px": 10, "height_px": 10},
            }
        )

