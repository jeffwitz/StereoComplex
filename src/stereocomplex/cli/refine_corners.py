from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from stereocomplex.api.corner_refinement import CharucoDetections, refine_charuco_corners


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


def build_charuco_from_meta(meta: dict[str, Any]):
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
    elif hasattr(aruco, "CharucoBoard_create"):  # pragma: no cover
        board = aruco.CharucoBoard_create(squares_x, squares_y, square_size, marker_size, dictionary)
    else:  # pragma: no cover
        raise RuntimeError("cv2.aruco does not expose CharucoBoard APIs in this build.")

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


def detect_view(
    cv2,
    aruco,
    dictionary,
    board,
    detector_params,
    aruco_detector,
    charuco_detector,
    img_gray: np.ndarray,
) -> CharucoDetections | None:
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

    if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
        return None

    marker_ids_arr = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
    marker_corners_arr = [np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in marker_corners]

    if charuco_ids is None or charuco_corners is None or len(charuco_ids) == 0:
        charuco_ids_arr = np.zeros((0,), dtype=np.int32)
        charuco_xy = np.zeros((0, 2), dtype=np.float64)
    else:
        charuco_ids_arr = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        charuco_xy = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
        # Match dataset pixel-center convention.
        charuco_xy = charuco_xy - 0.5

    return CharucoDetections(
        marker_ids=marker_ids_arr,
        marker_corners=marker_corners_arr,
        charuco_ids=charuco_ids_arr,
        charuco_xy=charuco_xy,
    )


def _dict_from_ids_xy(ids: np.ndarray, xy: np.ndarray) -> dict[int, np.ndarray]:
    return {int(i): np.asarray(p, dtype=np.float64) for i, p in zip(ids.tolist(), xy.tolist(), strict=True)}


def refine_dataset_scene(
    *,
    dataset_root: Path,
    split: str,
    scene: str,
    method: str,
    max_frames: int,
    tps_lam: float,
    huber_c: float,
    iters: int,
) -> dict[str, Any]:
    import cv2  # type: ignore

    scene_dir = Path(dataset_root) / str(split) / str(scene)
    meta = load_json(scene_dir / "meta.json")
    frames = load_frames(scene_dir)
    if max_frames and max_frames > 0:
        frames = frames[: int(max_frames)]

    cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector = build_charuco_from_meta(meta)

    results: list[dict[str, Any]] = []
    for fr in frames:
        fid = int(fr["frame_id"])
        entry: dict[str, Any] = {"frame_id": fid}
        for side in ("left", "right"):
            img_path = scene_dir / side / str(fr[side])
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            det = detect_view(cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector, img)
            if det is None:
                continue
            refined_xy = refine_charuco_corners(
                method=str(method),
                board=board,
                marker_ids=det.marker_ids,
                marker_corners=det.marker_corners,
                charuco_ids=det.charuco_ids,
                charuco_xy=det.charuco_xy,
                tps_lam=float(tps_lam),
                huber_c=float(huber_c),
                iters=int(iters),
            )
            entry[side] = {
                "charuco_ids": det.charuco_ids.astype(int).tolist(),
                "charuco_xy_raw": det.charuco_xy.tolist(),
                "charuco_xy_refined": refined_xy.tolist(),
                "n_markers": int(det.marker_ids.size),
                "n_charuco": int(det.charuco_ids.size),
            }
        results.append(entry)

    return {
        "schema_version": "stereocomplex.refined_corners.v0",
        "dataset_root": str(Path(dataset_root).resolve()),
        "split": str(split),
        "scene": str(scene),
        "method": str(method),
        "tps": {"lam": float(tps_lam), "huber_c": float(huber_c), "iters": int(iters)},
        "frames": results,
    }


def make_calibration_npz(
    *,
    refined: dict[str, Any],
    out_npz: Path,
) -> None:
    """
    Export a minimal NPZ to feed OpenCV calibration:

    - per frame: common ids (left/right) and their 3D board coordinates
    - image points in pixel-center convention (same as the rest of the repo)
    """
    dataset_root = Path(refined["dataset_root"])
    split = str(refined["split"])
    scene = str(refined["scene"])
    scene_dir = dataset_root / split / scene
    meta = load_json(scene_dir / "meta.json")

    import cv2.aruco as aruco  # type: ignore

    board_meta = meta["board"]
    dict_name = str(board_meta.get("aruco_dictionary", "DICT_4X4_1000"))
    dict_id = getattr(aruco, dict_name, None)
    if dict_id is None:
        raise ValueError(f"Unknown aruco_dictionary: {dict_name}")
    dictionary = aruco.getPredefinedDictionary(dict_id)
    board = aruco.CharucoBoard(
        (int(board_meta["squares_x"]), int(board_meta["squares_y"])),
        float(board_meta["square_size_mm"]),
        float(board_meta["marker_size_mm"]),
        dictionary,
    )
    chess3 = np.asarray(board.getChessboardCorners(), dtype=np.float64)  # (Nc,3)

    frame_ids: list[int] = []
    obj_pts: list[np.ndarray] = []
    uvL_pts: list[np.ndarray] = []
    uvR_pts: list[np.ndarray] = []

    for fr in refined["frames"]:
        fid = int(fr["frame_id"])
        if "left" not in fr or "right" not in fr:
            continue
        L = fr["left"]
        R = fr["right"]
        mapL = _dict_from_ids_xy(np.asarray(L["charuco_ids"], dtype=np.int32), np.asarray(L["charuco_xy_refined"], dtype=np.float64))
        mapR = _dict_from_ids_xy(np.asarray(R["charuco_ids"], dtype=np.int32), np.asarray(R["charuco_xy_refined"], dtype=np.float64))
        common = sorted(set(mapL).intersection(mapR))
        if len(common) < 6:
            continue
        uvL = np.stack([mapL[i] for i in common], axis=0).astype(np.float64)
        uvR = np.stack([mapR[i] for i in common], axis=0).astype(np.float64)
        obj = chess3[np.asarray(common, dtype=np.int32)].astype(np.float64)
        frame_ids.append(fid)
        obj_pts.append(obj)
        uvL_pts.append(uvL)
        uvR_pts.append(uvR)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        frame_id=np.asarray(frame_ids, dtype=np.int32),
        obj_pts_mm=np.asarray(obj_pts, dtype=object),
        uv_left_px=np.asarray(uvL_pts, dtype=object),
        uv_right_px=np.asarray(uvR_pts, dtype=object),
    )


def run_refine_corners(
    *,
    dataset_root: Path,
    split: str,
    scene: str,
    method: str,
    max_frames: int,
    tps_lam: float,
    huber_c: float,
    iters: int,
    out_json: Path,
    out_npz: Path | None,
) -> None:
    refined = refine_dataset_scene(
        dataset_root=dataset_root,
        split=split,
        scene=scene,
        method=method,
        max_frames=max_frames,
        tps_lam=tps_lam,
        huber_c=huber_c,
        iters=iters,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(refined, indent=2, sort_keys=True), encoding="utf-8")
    if out_npz is not None:
        make_calibration_npz(refined=refined, out_npz=out_npz)

