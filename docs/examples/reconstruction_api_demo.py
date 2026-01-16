"""
Reconstruction API demo (stereo central ray-field).

This script is meant to be:
- readable (heavily commented),
- runnable (no hidden imports),
- aligned with docs/RECONSTRUCTION_API.md.

It does:
1) load an exported stereo model (model.json + weights.npz),
2) detect ChArUco corners on a few frames (OpenCV),
3) triangulate 3D points with the API,
4) (optional) compare to GT 3D on synthetic datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from stereocomplex.api import load_stereo_central_rayfield


Side = Literal["left", "right"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frames(scene_dir: Path) -> list[dict[str, Any]]:
    frames_path = scene_dir / "frames.jsonl"
    frames: list[dict[str, Any]] = []
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        frames.append(json.loads(line))
    return frames


def summarize(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "rms": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    v = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(v.size),
        "rms": float(np.sqrt(np.mean(v * v))),
        "p50": float(np.quantile(v, 0.50)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(np.max(v)),
    }


def _dict_from_ids_xy(ids: np.ndarray, xy: np.ndarray) -> dict[int, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return {int(i): xy[k].astype(np.float64) for k, i in enumerate(ids.tolist())}


def _stack_for_ids(ids: list[int], mapping: dict[int, np.ndarray]) -> np.ndarray:
    return np.asarray([mapping[int(i)] for i in ids], dtype=np.float64).reshape(-1, 2)


def build_charuco_from_meta(meta: dict[str, Any]):
    """
    Minimal OpenCV Charuco detector setup (used only to get 2D correspondences).
    """
    import cv2  # type: ignore
    import cv2.aruco as aruco  # type: ignore

    board_meta = meta["board"]
    dict_name = str(board_meta.get("aruco_dictionary", "DICT_4X4_1000"))
    dict_id = getattr(aruco, dict_name, None)
    if dict_id is None:
        raise ValueError(f"Unknown aruco_dictionary: {dict_name}")
    dictionary = aruco.getPredefinedDictionary(dict_id)

    squares_x = int(board_meta["squares_x"])
    squares_y = int(board_meta["squares_y"])
    square_size = float(board_meta["square_size_mm"])
    marker_size = float(board_meta["marker_size_mm"])

    if hasattr(aruco, "CharucoBoard"):
        board = aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, dictionary)
    else:  # pragma: no cover
        board = aruco.CharucoBoard_create(squares_x, squares_y, square_size, marker_size, dictionary)

    detector_params = aruco.DetectorParameters()
    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 50
    detector_params.cornerRefinementMinAccuracy = 1e-3

    charuco_detector = None
    if hasattr(aruco, "CharucoDetector"):
        charuco_detector = aruco.CharucoDetector(board)
        if hasattr(charuco_detector, "setDetectorParameters"):
            charuco_detector.setDetectorParameters(detector_params)

    aruco_detector = None
    if charuco_detector is None and hasattr(aruco, "ArucoDetector"):
        aruco_detector = aruco.ArucoDetector(dictionary, detector_params)

    return cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector


def detect_charuco(
    cv2,
    aruco,
    dictionary,
    board,
    detector_params,
    aruco_detector,
    charuco_detector,
    img_gray: np.ndarray,
) -> dict[int, np.ndarray] | None:
    """
    Returns a mapping {corner_id -> (u,v)} in the dataset pixel-center convention.
    """
    if charuco_detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img_gray)
    else:
        if aruco_detector is not None:
            marker_corners, marker_ids, _rej = aruco_detector.detectMarkers(img_gray)
        else:  # pragma: no cover
            marker_corners, marker_ids, _rej = aruco.detectMarkers(img_gray, dictionary, parameters=detector_params)

        charuco_corners, charuco_ids = None, None
        if hasattr(aruco, "interpolateCornersCharuco") and marker_ids is not None and len(marker_ids) > 0:
            ret = aruco.interpolateCornersCharuco(marker_corners, marker_ids, img_gray, board)
            if ret is not None and len(ret) >= 2:
                if len(ret) == 3:
                    charuco_corners, charuco_ids, _ = ret
                elif len(ret) == 4:  # pragma: no cover
                    _, charuco_corners, charuco_ids, _ = ret

    if charuco_ids is None or charuco_corners is None or len(charuco_ids) == 0:
        return None

    # OpenCV often reports corners in a convention shifted by +0.5 px w.r.t.
    # our dataset convention (pixel centers at integer coordinates).
    charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    charuco_xy = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2) - 0.5
    return _dict_from_ids_xy(charuco_ids, charuco_xy)


def load_gt_charuco(scene_dir: Path) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, dict[Side, dict[int, np.ndarray]]]]:
    gt_path = scene_dir / "gt_charuco_corners.npz"
    if not gt_path.exists():
        return {}, {}
    gt = np.load(str(gt_path))
    gt_frame_id = gt["frame_id"].astype(np.int32).reshape(-1)
    gt_corner_id = gt["corner_id"].astype(np.int32).reshape(-1)
    gt_xyz = gt["XYZ_world_mm"].astype(np.float64).reshape(-1, 3)
    gt_uvL = gt["uv_left_px"].astype(np.float64).reshape(-1, 2)
    gt_uvR = gt["uv_right_px"].astype(np.float64).reshape(-1, 2)

    gt_xyz_by_frame: dict[int, dict[int, np.ndarray]] = {}
    gt_uv_by_frame: dict[int, dict[Side, dict[int, np.ndarray]]] = {}
    for fid in np.unique(gt_frame_id).tolist():
        mask = gt_frame_id == int(fid)
        ids = gt_corner_id[mask].tolist()
        xyz = gt_xyz[mask]
        uvL = gt_uvL[mask]
        uvR = gt_uvR[mask]
        gt_xyz_by_frame[int(fid)] = {int(i): xyz[k] for k, i in enumerate(ids)}
        gt_uv_by_frame[int(fid)] = {
            "left": {int(i): uvL[k] for k, i in enumerate(ids)},
            "right": {int(i): uvR[k] for k, i in enumerate(ids)},
        }
    return gt_xyz_by_frame, gt_uv_by_frame


def main() -> int:
    ap = argparse.ArgumentParser(description="Stereo reconstruction demo using the StereoComplex API.")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--scene", default="scene_0000")
    ap.add_argument("--model", type=Path, required=True, help="Directory containing model.json + weights.npz.")
    ap.add_argument("--max-frames", type=int, default=5, help="Frames to evaluate (0=all).")
    args = ap.parse_args()

    scene_dir = Path(args.dataset_root) / str(args.split) / str(args.scene)
    meta = load_json(scene_dir / "meta.json")
    frames = load_frames(scene_dir)
    if args.max_frames and args.max_frames > 0:
        frames = frames[: int(args.max_frames)]

    # 1) Load the stereo model once.
    model = load_stereo_central_rayfield(Path(args.model))

    # 2) Set up a ChArUco detector to produce 2D correspondences (u,v) in each image.
    cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector = build_charuco_from_meta(meta)

    # Optional: GT for error measurement (synthetic datasets only).
    gt_xyz_by_frame, _gt_uv_by_frame = load_gt_charuco(scene_dir)

    tri_err_mm: list[float] = []
    skew_mm: list[float] = []
    used_frames: list[int] = []
    n_corr: list[int] = []

    for fr in frames:
        fid = int(fr["frame_id"])

        # Load left/right images.
        imgL = cv2.imread(str(scene_dir / "left" / str(fr["left"])), cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(str(scene_dir / "right" / str(fr["right"])), cv2.IMREAD_GRAYSCALE)
        if imgL is None or imgR is None:
            continue

        # Detect ChArUco corners in both views.
        detL = detect_charuco(cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector, imgL)
        detR = detect_charuco(cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector, imgR)
        if detL is None or detR is None:
            continue

        common = sorted(set(detL).intersection(detR))
        if len(common) < 6:
            continue

        uvL = _stack_for_ids(common, detL)
        uvR = _stack_for_ids(common, detR)

        # 3) Reconstruct 3D points in the left camera frame.
        XYZ_hat_L, skew = model.triangulate(uvL, uvR)

        # 4) Optional GT error (requires gt_charuco_corners.npz).
        gt_xyz_map = gt_xyz_by_frame.get(fid, {})
        if gt_xyz_map:
            gt_common = [cid for cid in common if int(cid) in gt_xyz_map]
            if len(gt_common) >= 6:
                idx = {int(cid): k for k, cid in enumerate(common)}
                for cid in gt_common:
                    k = idx[int(cid)]
                    tri_err_mm.append(float(np.linalg.norm(XYZ_hat_L[k] - gt_xyz_map[int(cid)])))

        skew_mm.extend(np.asarray(skew, dtype=np.float64).tolist())
        used_frames.append(fid)
        n_corr.append(len(common))

    report = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "split": str(args.split),
        "scene": str(args.scene),
        "model_dir": str(Path(args.model).resolve()),
        "frames_used": used_frames,
        "n_corr_per_frame": summarize([float(x) for x in n_corr]),
        "triangulation_error_mm": summarize(tri_err_mm),
        "ray_skew_mm": summarize(skew_mm),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

