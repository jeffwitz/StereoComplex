from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


Side = Literal["left", "right"]


@dataclass(frozen=True)
class ViewDetections:
    marker_ids: np.ndarray  # (M,)
    marker_corners: list[np.ndarray]  # list of (4,2)
    charuco_ids: np.ndarray  # (K,)
    charuco_xy: np.ndarray  # (K,2) in dataset pixel-center convention


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


def focal_um_from_K(K: np.ndarray, pitch_um: float) -> dict[str, float]:
    return {
        "fx_um": float(K[0, 0] * pitch_um),
        "fy_um": float(K[1, 1] * pitch_um),
    }


def detect_view(
    cv2,
    aruco,
    dictionary,
    board,
    detector_params,
    aruco_detector,
    charuco_detector,
    img_gray: np.ndarray,
) -> ViewDetections | None:
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

    return ViewDetections(
        marker_ids=marker_ids_arr,
        marker_corners=marker_corners_arr,
        charuco_ids=charuco_ids_arr,
        charuco_xy=charuco_xy,
    )


def _dict_from_ids_xy(ids: np.ndarray, xy: np.ndarray) -> dict[int, np.ndarray]:
    return {int(i): np.asarray(p, dtype=np.float64) for i, p in zip(ids.tolist(), xy.tolist(), strict=True)}


def _stack_for_ids(ids: list[int], mapping: dict[int, np.ndarray]) -> np.ndarray:
    return np.asarray([mapping[int(i)] for i in ids], dtype=np.float64).reshape(-1, 2)


def calibrate_camera(obj_pts_list: list[np.ndarray], img_pts_list: list[np.ndarray], image_size: tuple[int, int]):
    import cv2  # type: ignore

    flags = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts_list,
        img_pts_list,
        image_size,
        None,
        None,
        flags=flags,
        criteria=criteria,
    )
    return float(rms), K, dist, rvecs, tvecs


def stereo_calibrate(
    obj_pts_list: list[np.ndarray],
    imgL_list: list[np.ndarray],
    imgR_list: list[np.ndarray],
    image_size: tuple[int, int],
    K1: np.ndarray,
    d1: np.ndarray,
    K2: np.ndarray,
    d2: np.ndarray,
):
    import cv2  # type: ignore

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    rms, K1o, d1o, K2o, d2o, R, T, E, F = cv2.stereoCalibrate(
        obj_pts_list,
        imgL_list,
        imgR_list,
        K1,
        d1,
        K2,
        d2,
        image_size,
        criteria=criteria,
        flags=flags,
    )
    return float(rms), K1o, d1o, K2o, d2o, R, T, E, F


def summarize(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "rms": float("nan"), "p95": float("nan"), "max": float("nan")}
    v = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(v.size),
        "rms": float(np.sqrt(np.mean(v * v))),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(np.max(v)),
    }


def summarize_dist(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "p05": float("nan"), "p50": float("nan"), "p95": float("nan")}
    v = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(v.size),
        "p05": float(np.quantile(v, 0.05)),
        "p50": float(np.quantile(v, 0.50)),
        "p95": float(np.quantile(v, 0.95)),
    }


def summarize_counts(vals: list[int]) -> dict[str, float]:
    if not vals:
        return {
            "n": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    v = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "p50": float(np.quantile(v, 0.50)),
        "p95": float(np.quantile(v, 0.95)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def rel_percent_abs(est: float, gt: float, *, eps: float = 1e-12) -> float:
    """
    Relative absolute error in percent.

    Returns NaN when gt is (near) zero to avoid meaningless percentages.
    """
    if not np.isfinite(gt) or abs(gt) <= eps:
        return float("nan")
    return float(100.0 * abs(est - gt) / abs(gt))


def triangulate_points(
    K1: np.ndarray,
    d1: np.ndarray,
    K2: np.ndarray,
    d2: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    uv1: np.ndarray,
    uv2: np.ndarray,
) -> np.ndarray:
    """
    Triangulate in the left camera frame using undistorted normalized coordinates.

    Returns: (N,3) in the same length unit as T (here: mm).
    """
    import cv2  # type: ignore

    uv1 = np.asarray(uv1, dtype=np.float64).reshape(-1, 2)
    uv2 = np.asarray(uv2, dtype=np.float64).reshape(-1, 2)
    p1 = cv2.undistortPoints(uv1.reshape(-1, 1, 2), K1, d1).reshape(-1, 2)
    p2 = cv2.undistortPoints(uv2.reshape(-1, 1, 2), K2, d2).reshape(-1, 2)

    P1 = np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    P2 = np.hstack([np.asarray(R, dtype=np.float64), np.asarray(T, dtype=np.float64).reshape(3, 1)])

    Xh = cv2.triangulatePoints(P1, P2, p1.T, p2.T)  # (4,N)
    X = (Xh[:3, :] / (Xh[3:4, :] + 1e-12)).T
    return X.astype(np.float64)


def rectify_points(
    cv2,
    *,
    uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R_rect: np.ndarray,
    P_rect: np.ndarray,
) -> np.ndarray:
    """
    Rectify points into pixel coordinates of the rectified images.

    Note: P returned by cv2.stereoRectify is (3,4); cv2.undistortPoints also
    accepts (3,3), so we pass the left 3x3 block.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    P3 = np.asarray(P_rect, dtype=np.float64)
    if P3.shape == (3, 4):
        P3 = P3[:, :3]
    out = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, dist, R=R_rect, P=P3).reshape(-1, 2)
    return out.astype(np.float64)


def skew_lines_distance(
    *,
    C2: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Distance between two 3D lines:
      L1(λ) = λ d1  (with origin at C1 = 0)
      L2(μ) = C2 + μ d2

    Returns per-point distances in the same units as C2.
    """
    C2 = np.asarray(C2, dtype=np.float64).reshape(3)
    d1 = np.asarray(d1, dtype=np.float64).reshape(-1, 3)
    d2 = np.asarray(d2, dtype=np.float64).reshape(-1, 3)
    n = np.cross(d1, d2)  # (N,3)
    n_norm = np.linalg.norm(n, axis=1)
    num = np.abs(n @ C2.reshape(3, 1)).reshape(-1)
    return (num / (n_norm + eps)).astype(np.float64)


def distortion_displacement_metrics(
    cv2,
    *,
    K_gt: np.ndarray,
    dist_gt: np.ndarray,
    dist_est: np.ndarray,
    image_size: tuple[int, int],
    n_angles: int = 180,
    radii_fracs: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9),
) -> dict[str, float]:
    """
    Compare distortion models in pixel space (physically meaningful for an image sensor).

    We sample ideal (undistorted) pixels on concentric circles, apply both distortion
    models using cv2.projectPoints, and compare the resulting distortion displacement
    vectors (in pixels).
    """
    w, h = image_size
    cx = float(K_gt[0, 2])
    cy = float(K_gt[1, 2])
    fx = float(K_gt[0, 0])
    fy = float(K_gt[1, 1])

    # Keep samples inside the image.
    rmax = max(1.0, min(cx, (w - 1) - cx, cy, (h - 1) - cy))
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False, dtype=np.float64)

    uv_list: list[np.ndarray] = []
    for rf in radii_fracs:
        r = float(rf) * rmax
        u = cx + r * np.cos(angles)
        v = cy + r * np.sin(angles)
        uv_list.append(np.stack([u, v], axis=1))
    uv = np.concatenate(uv_list, axis=0).astype(np.float64)  # (N,2) ideal pixels

    x = (uv[:, 0] - cx) / fx
    y = (uv[:, 1] - cy) / fy
    obj = np.stack([x, y, np.ones_like(x)], axis=1).reshape(-1, 1, 3).astype(np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)

    uv_gt, _ = cv2.projectPoints(obj, rvec, tvec, K_gt, dist_gt)
    uv_est, _ = cv2.projectPoints(obj, rvec, tvec, K_gt, dist_est)
    uv_gt = uv_gt.reshape(-1, 2)
    uv_est = uv_est.reshape(-1, 2)

    disp_gt = uv_gt - uv
    disp_est = uv_est - uv
    disp_err = disp_est - disp_gt

    gt_mag = np.linalg.norm(disp_gt, axis=1)
    err_mag = np.linalg.norm(disp_err, axis=1)

    gt_rms = float(np.sqrt(np.mean(gt_mag * gt_mag)))
    err_rms = float(np.sqrt(np.mean(err_mag * err_mag)))
    rel_rms_pct = float(100.0 * err_rms / (gt_rms + 1e-12))
    return {
        "gt_rms_px": gt_rms,
        "err_rms_px": err_rms,
        "err_rms_percent_of_gt": rel_rms_pct,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare OpenCV calibration using raw vs ray-field corrected corners.")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--scene", default="scene_0000")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit frames (0=all).")
    ap.add_argument("--out", type=Path, default=Path("paper/tables/opencv_calibration_rayfield.json"))
    ap.add_argument("--tps-lam", type=float, default=10.0)
    ap.add_argument("--tps-huber", type=float, default=3.0)
    ap.add_argument("--tps-iters", type=int, default=3)
    args = ap.parse_args()

    scene_dir = Path(args.dataset_root) / str(args.split) / str(args.scene)
    meta = load_json(scene_dir / "meta.json")
    frames = load_frames(scene_dir)
    if args.max_frames and args.max_frames > 0:
        frames = frames[: int(args.max_frames)]

    cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector = build_charuco_from_meta(meta)

    # Marker-id -> board-plane marker corners.
    board_ids = np.asarray(board.getIds(), dtype=np.int32).reshape(-1)
    board_obj = board.getObjPoints()
    id_to_obj2 = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

    chess3 = np.asarray(board.getChessboardCorners(), dtype=np.float64)  # (Nc,3)
    chess2 = chess3[:, :2]
    all_ids = np.arange(chess3.shape[0], dtype=np.int32)

    image_size = (int(meta["stereo"]["left"]["image"]["width_px"]), int(meta["stereo"]["left"]["image"]["height_px"]))

    # Import ray-field correction.
    from stereocomplex.eval.charuco_detection import _camera_params_from_meta, _predict_points_rayfield_tps_robust  # noqa: PLC0415

    # Ground-truth (synthetic datasets only): derive pinhole K from meta + f_um, and Brown distortion if present.
    sim = meta.get("sim_params", {})
    f_um = float(sim.get("f_um", float("nan")))
    dist_model = str(sim.get("distortion_model", "none"))
    dL_gt = sim.get("distortion_left", {}) if dist_model == "brown" else {}
    dR_gt = sim.get("distortion_right", {}) if dist_model == "brown" else {}
    K_L_gt, dist_L_gt = _camera_params_from_meta(meta["stereo"]["left"], f_um=f_um, brown=dL_gt)
    K_R_gt, dist_R_gt = _camera_params_from_meta(meta["stereo"]["right"], f_um=f_um, brown=dR_gt)
    baseline_gt = float(sim.get("baseline_mm", float("nan")))

    # Ground-truth 3D points are given in the NPZ, in the same convention as uv_left_px/uv_right_px.
    gt_npz = np.load(str(scene_dir / "gt_charuco_corners.npz"))
    gt_frame_id = gt_npz["frame_id"].astype(np.int32).reshape(-1)
    gt_corner_id = gt_npz["corner_id"].astype(np.int32).reshape(-1)
    gt_xyz = gt_npz["XYZ_world_mm"].astype(np.float64).reshape(-1, 3)
    gt_uv_left = gt_npz["uv_left_px"].astype(np.float64).reshape(-1, 2)
    gt_uv_right = gt_npz["uv_right_px"].astype(np.float64).reshape(-1, 2)
    gt_by_frame: dict[int, dict[int, np.ndarray]] = {}
    gt_uv_by_frame: dict[int, dict[Side, dict[int, np.ndarray]]] = {}
    for fid in np.unique(gt_frame_id).tolist():
        mask = gt_frame_id == int(fid)
        ids = gt_corner_id[mask]
        xyz = gt_xyz[mask]
        gt_by_frame[int(fid)] = {int(i): xyz[k] for k, i in enumerate(ids.tolist())}
        uvL = gt_uv_left[mask]
        uvR = gt_uv_right[mask]
        gt_uv_by_frame[int(fid)] = {
            "left": {int(i): uvL[k].astype(np.float64) for k, i in enumerate(ids.tolist())},
            "right": {int(i): uvR[k].astype(np.float64) for k, i in enumerate(ids.tolist())},
        }

    def rayfield_predict(
        marker_ids: np.ndarray,
        marker_corners: list[np.ndarray],
        target_ids: np.ndarray,
    ) -> dict[int, np.ndarray] | None:
        """
        Predict ChArUco corners only for IDs that were actually observed by OpenCV.

        Important: for calibration, we must not hallucinate unobserved points.
        """
        obj_pts: list[np.ndarray] = []
        img_pts: list[np.ndarray] = []
        for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
            o = id_to_obj2.get(int(mid))
            if o is None or mc.shape != (4, 2) or o.shape != (4, 2):
                continue
            obj_pts.append(o)
            img_pts.append(mc)
        if not obj_pts:
            return None
        obj_xy = np.concatenate(obj_pts, axis=0)
        img_uv = np.concatenate(img_pts, axis=0)
        target_ids = np.asarray(target_ids, dtype=np.int32).reshape(-1)
        if target_ids.size == 0:
            return {}
        target_xy = chess2[target_ids]
        pred = _predict_points_rayfield_tps_robust(
            obj_xy,
            img_uv,
            target_xy,
            lam=float(args.tps_lam),
            huber_c=float(args.tps_huber),
            iters=int(args.tps_iters),
        )
        return _dict_from_ids_xy(target_ids, pred)

    methods = ["raw", "rayfield_tps_robust"]
    report: dict[str, Any] = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "split": str(args.split),
        "scene": str(args.scene),
        "tps": {"lam": float(args.tps_lam), "huber_c": float(args.tps_huber), "iters": int(args.tps_iters)},
        "gt": {
            "baseline_mm": baseline_gt,
            "left": {"K": K_L_gt.tolist(), "dist": dist_L_gt.reshape(-1).tolist()},
            "right": {"K": K_R_gt.tolist(), "dist": dist_R_gt.reshape(-1).tolist()},
        },
        "methods": {},
    }

    for method in methods:
        objL: list[np.ndarray] = []
        imgL: list[np.ndarray] = []
        objR: list[np.ndarray] = []
        imgR: list[np.ndarray] = []
        objS: list[np.ndarray] = []
        imgSL: list[np.ndarray] = []
        imgSR: list[np.ndarray] = []

        used_frames_left: set[int] = set()
        used_frames_right: set[int] = set()
        used_frames_stereo: set[int] = set()
        n_corners_left: list[int] = []
        n_corners_right: list[int] = []
        n_corners_stereo: list[int] = []

        n_frames_used = 0

        for fr in frames:
            fid = int(fr["frame_id"])
            pts_by_side: dict[Side, dict[int, np.ndarray]] = {}
            markers_by_side: dict[Side, tuple[np.ndarray, list[np.ndarray]]] = {}

            for side in ("left", "right"):
                img_path = scene_dir / side / str(fr[side])
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                det = detect_view(cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector, img)
                if det is None:
                    continue
                markers_by_side[side] = (det.marker_ids, det.marker_corners)

                if method == "raw":
                    pts_by_side[side] = _dict_from_ids_xy(det.charuco_ids, det.charuco_xy)
                else:
                    pred_map = rayfield_predict(det.marker_ids, det.marker_corners, det.charuco_ids)
                    if pred_map is None:
                        continue
                    pts_by_side[side] = pred_map

            if "left" not in pts_by_side or "right" not in pts_by_side:
                continue

            # Per-camera calibration points.
            for side, obj_list, img_list in [("left", objL, imgL), ("right", objR, imgR)]:
                pts = pts_by_side[side]
                ids = sorted(pts.keys())
                if len(ids) < 6:
                    continue
                img_xy = _stack_for_ids(ids, pts)
                obj_xyz = chess3[np.asarray(ids, dtype=np.int32)]
                obj_list.append(obj_xyz.reshape(-1, 1, 3).astype(np.float32))
                img_list.append(img_xy.reshape(-1, 1, 2).astype(np.float32))
                if side == "left":
                    used_frames_left.add(fid)
                    n_corners_left.append(int(len(ids)))
                else:
                    used_frames_right.add(fid)
                    n_corners_right.append(int(len(ids)))

            # Stereo calibration requires corresponding points; use intersection.
            common = sorted(set(pts_by_side["left"].keys()).intersection(pts_by_side["right"].keys()))
            if len(common) >= 6:
                img_xy_L = _stack_for_ids(common, pts_by_side["left"])
                img_xy_R = _stack_for_ids(common, pts_by_side["right"])
                obj_xyz = chess3[np.asarray(common, dtype=np.int32)]
                objS.append(obj_xyz.reshape(-1, 1, 3).astype(np.float32))
                imgSL.append(img_xy_L.reshape(-1, 1, 2).astype(np.float32))
                imgSR.append(img_xy_R.reshape(-1, 1, 2).astype(np.float32))
                used_frames_stereo.add(fid)
                n_corners_stereo.append(int(len(common)))

            n_frames_used += 1

        entry: dict[str, Any] = {
            "n_frames": int(n_frames_used),
            "n_views_left": int(len(imgL)),
            "n_views_right": int(len(imgR)),
            "n_views_stereo": int(len(objS)),
            "view_stats": {
                "left": {
                    "frame_ids": sorted(used_frames_left),
                    "n_corners": summarize_counts(n_corners_left),
                },
                "right": {
                    "frame_ids": sorted(used_frames_right),
                    "n_corners": summarize_counts(n_corners_right),
                },
                "stereo": {
                    "frame_ids": sorted(used_frames_stereo),
                    "n_corners": summarize_counts(n_corners_stereo),
                },
            },
        }

        if len(objL) >= 3 and len(objR) >= 3:
            rmsL, KL, dL, _rvL, _tvL = calibrate_camera(objL, imgL, image_size)
            rmsR, KR, dR, _rvR, _tvR = calibrate_camera(objR, imgR, image_size)
            # Compare to GT (synthetic).
            def k_err(K_est: np.ndarray, K_gt: np.ndarray) -> dict[str, float]:
                return {
                    "fx": float(K_est[0, 0] - K_gt[0, 0]),
                    "fy": float(K_est[1, 1] - K_gt[1, 1]),
                    "cx": float(K_est[0, 2] - K_gt[0, 2]),
                    "cy": float(K_est[1, 2] - K_gt[1, 2]),
                }

            def d_err(d_est: np.ndarray, d_gt: np.ndarray) -> list[float]:
                de = d_est.reshape(-1).astype(np.float64)
                dg = d_gt.reshape(-1).astype(np.float64)
                n = min(int(de.size), int(dg.size))
                return (de[:n] - dg[:n]).tolist()

            def k_percent(K_est: np.ndarray, K_gt: np.ndarray) -> dict[str, float]:
                return {
                    "fx": rel_percent_abs(float(K_est[0, 0]), float(K_gt[0, 0])),
                    "fy": rel_percent_abs(float(K_est[1, 1]), float(K_gt[1, 1])),
                }

            def dist_percent(d_est: np.ndarray, d_gt: np.ndarray) -> dict[str, float]:
                dist_names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
                de = d_est.reshape(-1).astype(np.float64)
                dg = d_gt.reshape(-1).astype(np.float64)
                n = min(int(de.size), int(dg.size))
                out: dict[str, float] = {}
                for i in range(n):
                    name = dist_names[i] if i < len(dist_names) else f"d{i}"
                    out[name] = rel_percent_abs(float(de[i]), float(dg[i]))
                denom = float(np.linalg.norm(dg[:n]))
                out["norm_l2"] = float(100.0 * np.linalg.norm(de[:n] - dg[:n]) / (denom + 1e-12))
                return out

            entry["mono"] = {
                "left": {"rms": rmsL, "K": KL.tolist(), "dist": dL.reshape(-1).tolist()},
                "right": {"rms": rmsR, "K": KR.tolist(), "dist": dR.reshape(-1).tolist()},
                "delta_vs_gt": {
                    "left": {"K": k_err(KL, K_L_gt), "dist": d_err(dL, dist_L_gt)},
                    "right": {"K": k_err(KR, K_R_gt), "dist": d_err(dR, dist_R_gt)},
                },
                "percent_vs_gt": {
                    "left": {"K": k_percent(KL, K_L_gt), "dist": dist_percent(dL, dist_L_gt)},
                    "right": {"K": k_percent(KR, K_R_gt), "dist": dist_percent(dR, dist_R_gt)},
                },
                "distortion_displacement_vs_gt": {
                    "left": distortion_displacement_metrics(
                        cv2,
                        K_gt=K_L_gt,
                        dist_gt=dist_L_gt,
                        dist_est=dL,
                        image_size=image_size,
                    ),
                    "right": distortion_displacement_metrics(
                        cv2,
                        K_gt=K_R_gt,
                        dist_gt=dist_R_gt,
                        dist_est=dR,
                        image_size=image_size,
                    ),
                },
                "focal_um": {
                    "left": focal_um_from_K(KL, float(meta["stereo"]["left"]["sensor"]["pixel_pitch_um"])),
                    "right": focal_um_from_K(KR, float(meta["stereo"]["right"]["sensor"]["pixel_pitch_um"])),
                    "gt_left": focal_um_from_K(K_L_gt, float(meta["stereo"]["left"]["sensor"]["pixel_pitch_um"])),
                    "gt_right": focal_um_from_K(K_R_gt, float(meta["stereo"]["right"]["sensor"]["pixel_pitch_um"])),
                },
            }

            if len(objS) >= 3:
                rmsS, _KL, _dL, _KR, _dR, R, T, _E, _F = stereo_calibrate(objS, imgSL, imgSR, image_size, KL, dL, KR, dR)
                baseline_est = float(np.linalg.norm(T))
                baseline_delta = float(baseline_est - baseline_gt) if np.isfinite(baseline_gt) else float("nan")

                # Rectification transforms from the estimated stereo model.
                try:
                    R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
                        KL,
                        dL,
                        KR,
                        dR,
                        image_size,
                        R,
                        T,
                        flags=cv2.CALIB_ZERO_DISPARITY,
                        alpha=0.0,
                    )
                except Exception:  # pragma: no cover
                    R1 = R2 = P1 = P2 = None

                # Baseline error expressed as an equivalent disparity error (px) at the GT depths.
                # delta_d(px) = fx(px) * delta_B(mm) / Z(mm)
                baseline_px: list[float] = []
                tri_err_mm: list[float] = []
                tri_err_rel_z_pct: list[float] = []
                z_used_mm: list[float] = []
                vdisp_meas_px: list[float] = []
                vdisp_gt_px: list[float] = []
                disp_err_meas_px: list[float] = []
                ray_skew_mm: list[float] = []

                # Rebuild per-frame 2D correspondences for triangulation evaluation.
                for fr in frames:
                    fid = int(fr["frame_id"])
                    gt_xyz_map = gt_by_frame.get(fid, {})
                    if not gt_xyz_map:
                        continue
                    gt_uv_map = gt_uv_by_frame.get(fid, {})

                    # Recompute 2D points for this frame (same logic as above, but we need both views here).
                    pts_frame: dict[Side, dict[int, np.ndarray]] = {}
                    for side in ("left", "right"):
                        img_path = scene_dir / side / str(fr[side])
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        det = detect_view(cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector, img)
                        if det is None:
                            continue
                        if method == "raw":
                            pts_frame[side] = _dict_from_ids_xy(det.charuco_ids, det.charuco_xy)
                        else:
                            pred_map = rayfield_predict(det.marker_ids, det.marker_corners, det.charuco_ids)
                            if pred_map is None:
                                continue
                            pts_frame[side] = pred_map

                    if "left" not in pts_frame or "right" not in pts_frame:
                        continue

                    common = sorted(set(pts_frame["left"].keys()).intersection(pts_frame["right"].keys()))
                    if len(common) < 6:
                        continue

                    uvL = _stack_for_ids(common, pts_frame["left"])
                    uvR = _stack_for_ids(common, pts_frame["right"])

                    # Ray skew distance: how far the two reconstructed rays are from intersecting (in mm).
                    p1 = cv2.undistortPoints(uvL.reshape(-1, 1, 2), KL, dL).reshape(-1, 2)
                    p2 = cv2.undistortPoints(uvR.reshape(-1, 1, 2), KR, dR).reshape(-1, 2)
                    dL_ray = np.concatenate([p1, np.ones((p1.shape[0], 1), dtype=np.float64)], axis=1)
                    dR_ray = np.concatenate([p2, np.ones((p2.shape[0], 1), dtype=np.float64)], axis=1)
                    dR_ray = (R.T @ dR_ray.T).T  # rotate into left frame
                    dL_ray /= np.linalg.norm(dL_ray, axis=1, keepdims=True) + 1e-12
                    dR_ray /= np.linalg.norm(dR_ray, axis=1, keepdims=True) + 1e-12
                    C2 = (-R.T @ T.reshape(3, 1)).reshape(3)  # right camera center in left frame
                    ray_skew_mm.extend(skew_lines_distance(C2=C2, d1=dL_ray, d2=dR_ray).tolist())

                    if R1 is not None and R2 is not None and P1 is not None and P2 is not None:
                        uvLr = rectify_points(cv2, uv=uvL, K=KL, dist=dL, R_rect=R1, P_rect=P1)
                        uvRr = rectify_points(cv2, uv=uvR, K=KR, dist=dR, R_rect=R2, P_rect=P2)
                        idx_of = {int(cid): k for k, cid in enumerate(common)}
                        for k in range(len(common)):
                            vdisp_meas_px.append(float(abs(uvLr[k, 1] - uvRr[k, 1])))

                        if gt_uv_map:
                            gtL_map = gt_uv_map.get("left", {})
                            gtR_map = gt_uv_map.get("right", {})
                            gt_common = [int(cid) for cid in common if int(cid) in gtL_map and int(cid) in gtR_map]
                            if gt_common:
                                gt_uvL = _stack_for_ids(gt_common, gtL_map)
                                gt_uvR = _stack_for_ids(gt_common, gtR_map)
                                gt_uvLr = rectify_points(cv2, uv=gt_uvL, K=KL, dist=dL, R_rect=R1, P_rect=P1)
                                gt_uvRr = rectify_points(cv2, uv=gt_uvR, K=KR, dist=dR, R_rect=R2, P_rect=P2)
                                for j, cid in enumerate(gt_common):
                                    vdisp_gt_px.append(float(abs(gt_uvLr[j, 1] - gt_uvRr[j, 1])))
                                    k = idx_of.get(int(cid))
                                    if k is None:
                                        continue
                                    d_meas = float(uvLr[k, 0] - uvRr[k, 0])
                                    d_gt = float(gt_uvLr[j, 0] - gt_uvRr[j, 0])
                                    disp_err_meas_px.append(float(abs(d_meas - d_gt)))

                    # Triangulate with current stereo calibration.
                    X = triangulate_points(KL, dL, KR, dR, R, T, uvL, uvR)
                    for k, cid in enumerate(common):
                        gtX = gt_xyz_map.get(int(cid))
                        if gtX is None:
                            continue
                        e = float(np.linalg.norm(X[k] - gtX))
                        tri_err_mm.append(e)
                        Z = float(gtX[2])
                        if Z > 1e-6:
                            tri_err_rel_z_pct.append(float(100.0 * e / Z))
                            z_used_mm.append(Z)

                        if np.isfinite(baseline_delta) and Z > 1e-6:
                            baseline_px.append(float(abs(KL[0, 0] * baseline_delta / Z)))

                entry["stereo"] = {
                    "rms": rmsS,
                    "R": R.tolist(),
                    "T": T.reshape(-1).tolist(),
                    "baseline_est_mm": baseline_est,
                    "baseline_delta_mm": baseline_delta,
                    "baseline_delta_px": summarize(baseline_px),
                    "depth_mm": summarize_dist(z_used_mm),
                    "triangulation_error_mm": summarize(tri_err_mm),
                    "triangulation_error_rel_z_percent": summarize(tri_err_rel_z_pct),
                    "rectification": {
                        "vertical_disparity_measured_px": summarize(vdisp_meas_px),
                        "vertical_disparity_gt_px": summarize(vdisp_gt_px),
                        "disparity_error_measured_px": summarize(disp_err_meas_px),
                    },
                    "ray_skew_distance_mm": summarize(ray_skew_mm),
                }
        report["methods"][method] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(json.dumps(report["methods"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
