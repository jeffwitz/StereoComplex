from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from stereocomplex.sim.dataset_validate import validate_dataset


def _write_manifest(root: Path) -> None:
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": "stereocomplex.dataset.v0"}, indent=2),
        encoding="utf-8",
    )


def _write_min_scene(scene_dir: Path, *, w: int = 16, h: int = 12) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "left").mkdir(parents=True, exist_ok=True)
    (scene_dir / "right").mkdir(parents=True, exist_ok=True)

    view = {
        "schema_version": "stereocomplex.meta.v0",
        "sensor": {"pixel_pitch_um": 3.45, "binning_xy": [1, 1]},
        "preprocess": {"crop_xywh_px": [0, 0, w, h], "resize_xy": [1.0, 1.0]},
        "image": {"width_px": w, "height_px": h, "bit_depth": 8, "gamma": 1.0},
    }
    meta = {
        "schema_version": "stereocomplex.dataset.v0",
        "stereo": {"left": view, "right": view},
        "board": {
            "type": "charuco",
            "square_size_mm": 1.0,
            "marker_size_mm": 0.8,
            "squares_x": 3,
            "squares_y": 3,
            "aruco_dictionary": "DICT_4X4_1000",
        },
    }
    (scene_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    fr = {"frame_id": 0, "left": "000000.png", "right": "000000.png"}
    (scene_dir / "frames.jsonl").write_text(json.dumps(fr) + "\n", encoding="utf-8")

    img = Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode="L")
    img.save(scene_dir / "left" / "000000.png")
    img.save(scene_dir / "right" / "000000.png")

    np.savez(
        scene_dir / "gt_points.npz",
        frame_id=np.asarray([0], dtype=np.int32),
        XYZ_world_mm=np.asarray([[0.0, 0.0, 10.0]], dtype=np.float64),
        uv_left_px=np.asarray([[0.0, 0.0]], dtype=np.float64),
        uv_right_px=np.asarray([[0.0, 0.0]], dtype=np.float64),
    )


def test_validate_dataset_root(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    root.mkdir()
    _write_manifest(root)
    _write_min_scene(root / "train" / "scene_0000")
    validate_dataset(root)


def test_validate_dataset_collection(tmp_path: Path) -> None:
    # Root without manifest.json but containing datasets (e.g. compression sweeps).
    collection = tmp_path / "collection"
    collection.mkdir()
    for name in ("a", "b"):
        ds = collection / name
        ds.mkdir()
        _write_manifest(ds)
        _write_min_scene(ds / "train" / "scene_0000")
    validate_dataset(collection)

