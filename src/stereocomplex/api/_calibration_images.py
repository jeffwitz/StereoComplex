from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from stereocomplex.api._calibration_types import CharucoBoardSpec, StereoImagePair
from stereocomplex.core.image_io import load_gray_u8


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_frames(scene_dir: Path) -> list[dict]:
    frames_path = scene_dir / "frames.jsonl"
    frames: list[dict] = []
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        frames.append(json.loads(line))
    return frames


def _ensure_gray_u8(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        return load_gray_u8(image)

    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[..., 0]
        else:
            arr = np.mean(arr[..., :3], axis=2)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _normalize_image_pairs(
    image_pairs: Sequence[StereoImagePair | tuple[str | Path, str | Path]],
) -> list[StereoImagePair]:
    out: list[StereoImagePair] = []
    for k, pair in enumerate(image_pairs):
        if isinstance(pair, StereoImagePair):
            fid = int(pair.frame_id) if pair.frame_id is not None else int(k)
            out.append(
                StereoImagePair(
                    left_path=Path(pair.left_path), right_path=Path(pair.right_path), frame_id=fid
                )
            )
            continue
        left_path, right_path = pair
        out.append(
            StereoImagePair(left_path=Path(left_path), right_path=Path(right_path), frame_id=int(k))
        )
    return out


def _sorted_image_paths(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in exts)


def _image_pairs_from_dirs(
    left_dir: str | Path, right_dir: str | Path, *, max_pairs: int = 0
) -> list[StereoImagePair]:
    left_paths = _sorted_image_paths(Path(left_dir))
    right_paths = _sorted_image_paths(Path(right_dir))
    if not left_paths or not right_paths:
        raise FileNotFoundError("no images found in left_dir/right_dir")
    if len(left_paths) != len(right_paths):
        raise ValueError("left_dir and right_dir must contain the same number of images")
    if max_pairs and max_pairs > 0:
        left_paths = left_paths[: int(max_pairs)]
        right_paths = right_paths[: int(max_pairs)]
    return [
        StereoImagePair(left_path=left_path, right_path=right_path, frame_id=k)
        for k, (left_path, right_path) in enumerate(zip(left_paths, right_paths, strict=True))
    ]


def _image_pairs_from_dataset(
    *,
    dataset_root: str | Path,
    split: str,
    scene: str,
    max_frames: int = 0,
) -> tuple[CharucoBoardSpec, list[StereoImagePair]]:
    scene_dir = Path(dataset_root) / str(split) / str(scene)
    meta = _load_json(scene_dir / "meta.json")
    board = CharucoBoardSpec.from_meta(meta)
    frames = _load_frames(scene_dir)
    if max_frames and max_frames > 0:
        frames = frames[: int(max_frames)]
    pairs = [
        StereoImagePair(
            left_path=scene_dir / "left" / str(frame["left"]),
            right_path=scene_dir / "right" / str(frame["right"]),
            frame_id=int(frame["frame_id"]),
        )
        for frame in frames
    ]
    return board, pairs
