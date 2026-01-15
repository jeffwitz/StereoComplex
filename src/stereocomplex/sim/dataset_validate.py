from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from stereocomplex.meta import MetaValidationError, parse_view_meta


def validate_dataset(dataset_root: Path) -> None:
    dataset_root = dataset_root.resolve()
    manifest = dataset_root / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}")

    meta = json.loads(manifest.read_text(encoding="utf-8"))
    if meta.get("schema_version") != "stereocomplex.dataset.v0":
        raise ValueError("manifest.json schema_version must be stereocomplex.dataset.v0")

    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            _validate_scene(scene_dir)


def _validate_scene(scene_dir: Path) -> None:
    meta_path = scene_dir / "meta.json"
    frames_path = scene_dir / "frames.jsonl"
    gt_path = scene_dir / "gt_points.npz"
    left_dir = scene_dir / "left"
    right_dir = scene_dir / "right"

    for p in (meta_path, frames_path, gt_path, left_dir, right_dir):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    scene_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if scene_meta.get("schema_version") != "stereocomplex.dataset.v0":
        raise ValueError(f"{meta_path} schema_version must be stereocomplex.dataset.v0")

    try:
        left_view = parse_view_meta(scene_meta["stereo"]["left"])
        right_view = parse_view_meta(scene_meta["stereo"]["right"])
    except KeyError as e:
        raise ValueError(f"{meta_path} missing key: {e}") from e
    except MetaValidationError as e:
        raise ValueError(f"{meta_path} invalid view meta: {e}") from e

    _validate_board_meta(scene_meta.get("board", {}), meta_path)

    npz = np.load(gt_path)
    for k in ("frame_id", "XYZ_world_mm", "uv_left_px", "uv_right_px"):
        if k not in npz:
            raise ValueError(f"{gt_path} missing key: {k}")

    n = int(npz["frame_id"].shape[0])
    if npz["XYZ_world_mm"].shape != (n, 3):
        raise ValueError("XYZ_world_mm must be (N,3)")
    if npz["uv_left_px"].shape != (n, 2) or npz["uv_right_px"].shape != (n, 2):
        raise ValueError("uv arrays must be (N,2)")

    # Validate that referenced frames exist.
    frames = frames_path.read_text(encoding="utf-8").splitlines()
    if not frames:
        raise ValueError(f"{frames_path} is empty")
    for line in frames:
        obj = json.loads(line)
        lf = left_dir / obj["left"]
        rf = right_dir / obj["right"]
        if not lf.exists() or not rf.exists():
            raise FileNotFoundError(f"Missing frame image: {lf} or {rf}")

    # Spot-check image sizes against meta (first frame only).
    first = json.loads(frames[0])
    lf0 = left_dir / first["left"]
    rf0 = right_dir / first["right"]
    with Image.open(lf0) as im:
        if im.size != (left_view.image.width_px, left_view.image.height_px):
            raise ValueError(f"{lf0} size {im.size} != meta {(left_view.image.width_px, left_view.image.height_px)}")
    with Image.open(rf0) as im:
        if im.size != (right_view.image.width_px, right_view.image.height_px):
            raise ValueError(f"{rf0} size {im.size} != meta {(right_view.image.width_px, right_view.image.height_px)}")


def _validate_board_meta(board: dict, meta_path: Path) -> None:
    if not isinstance(board, dict):
        raise ValueError(f"{meta_path} board must be an object")
    if "type" not in board:
        raise ValueError(f"{meta_path} board.type is required")
    if board["type"] == "texture_grid":
        for k in ("square_size_mm", "cols", "rows"):
            if k not in board:
                raise ValueError(f"{meta_path} board.{k} is required for texture_grid")
        if float(board["square_size_mm"]) <= 0:
            raise ValueError(f"{meta_path} board.square_size_mm must be > 0")
        if int(board["cols"]) <= 0 or int(board["rows"]) <= 0:
            raise ValueError(f"{meta_path} board cols/rows must be > 0")
    elif board["type"] == "charuco":
        for k in ("square_size_mm", "marker_size_mm", "squares_x", "squares_y", "aruco_dictionary"):
            if k not in board:
                raise ValueError(f"{meta_path} board.{k} is required for charuco")
        if float(board["square_size_mm"]) <= 0 or float(board["marker_size_mm"]) <= 0:
            raise ValueError(f"{meta_path} board square/marker sizes must be > 0")
        if float(board["marker_size_mm"]) >= float(board["square_size_mm"]):
            raise ValueError(f"{meta_path} board.marker_size_mm must be < board.square_size_mm")
        if int(board["squares_x"]) <= 0 or int(board["squares_y"]) <= 0:
            raise ValueError(f"{meta_path} board squares_x/squares_y must be > 0")
    else:
        raise ValueError(f"{meta_path} board.type unsupported: {board['type']}")
