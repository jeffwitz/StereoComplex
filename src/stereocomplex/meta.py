from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict


class MetaValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SensorMeta:
    pixel_pitch_um: float
    binning_xy: tuple[int, int]


@dataclass(frozen=True)
class PreprocessMeta:
    crop_xywh_px: tuple[int, int, int, int]
    resize_xy: tuple[float, float]


@dataclass(frozen=True)
class ImageMeta:
    width_px: int
    height_px: int
    bit_depth: int = 8
    gamma: float = 1.0


@dataclass(frozen=True)
class ViewMeta:
    schema_version: str
    sensor: SensorMeta
    preprocess: PreprocessMeta
    image: ImageMeta


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise MetaValidationError(msg)


def load_view_meta(path: Path) -> ViewMeta:
    data = json.loads(path.read_text(encoding="utf-8"))
    return parse_view_meta(data)


def parse_view_meta(data: dict[str, Any]) -> ViewMeta:
    schema_version = data.get("schema_version")
    _require(schema_version == "stereocomplex.meta.v0", "schema_version must be stereocomplex.meta.v0")

    sensor = data.get("sensor", {})
    preprocess = data.get("preprocess", {})
    image = data.get("image", {})

    pitch_raw = sensor.get("pixel_pitch_um")
    _require(pitch_raw is not None, "sensor.pixel_pitch_um is required")
    pitch_um = float(pitch_raw)
    _require(pitch_um > 0.0, "sensor.pixel_pitch_um must be > 0")

    binning = sensor.get("binning_xy", [1, 1])
    _require(isinstance(binning, (list, tuple)) and len(binning) == 2, "sensor.binning_xy must be [bx,by]")
    bx, by = int(binning[0]), int(binning[1])
    _require(bx >= 1 and by >= 1, "sensor.binning_xy values must be >= 1")

    crop = preprocess.get("crop_xywh_px")
    _require(
        isinstance(crop, (list, tuple)) and len(crop) == 4,
        "preprocess.crop_xywh_px must be [x,y,w,h] in binned sensor pixels",
    )
    crop_x, crop_y, crop_w, crop_h = (int(c) for c in crop)
    _require(crop_w > 0 and crop_h > 0, "crop w/h must be > 0")

    resize = preprocess.get("resize_xy", [1.0, 1.0])
    _require(isinstance(resize, (list, tuple)) and len(resize) == 2, "preprocess.resize_xy must be [sx,sy]")
    sx, sy = float(resize[0]), float(resize[1])
    _require(sx > 0 and sy > 0, "resize factors must be > 0")

    w_raw = image.get("width_px")
    h_raw = image.get("height_px")
    _require(w_raw is not None and h_raw is not None, "image.width_px and image.height_px are required")
    w = int(w_raw)
    h = int(h_raw)
    _require(w > 0 and h > 0, "image.width_px and image.height_px must be > 0")

    bit_depth = int(image.get("bit_depth", 8))
    gamma = float(image.get("gamma", 1.0))
    _require(bit_depth in (8, 10, 12, 14, 16), "image.bit_depth must be a common integer bit depth")
    _require(gamma > 0.0, "image.gamma must be > 0")

    # Minimal consistency check: crop+resize should match delivered image size (rounded).
    exp_w = int(round(crop_w * sx))
    exp_h = int(round(crop_h * sy))
    _require((exp_w, exp_h) == (w, h), f"image size must match crop*resize: expected {(exp_w, exp_h)}")

    return ViewMeta(
        schema_version=schema_version,
        sensor=SensorMeta(pixel_pitch_um=pitch_um, binning_xy=(bx, by)),
        preprocess=PreprocessMeta(crop_xywh_px=(crop_x, crop_y, crop_w, crop_h), resize_xy=(sx, sy)),
        image=ImageMeta(width_px=w, height_px=h, bit_depth=bit_depth, gamma=gamma),
    )
