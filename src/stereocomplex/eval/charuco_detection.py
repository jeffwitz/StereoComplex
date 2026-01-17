from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ErrorStats:
    n_matched: int
    rms_px: float
    mean_px: float
    p50_px: float
    p95_px: float
    max_px: float
    mean_dx_px: float
    mean_dy_px: float
    rms_dx_px: float
    rms_dy_px: float


def eval_charuco_detection(
    dataset_root: Path,
    write_json: bool = True,
    method: str = "charuco",
    refine: str = "none",
    tensor_sigma: float = 1.5,
    search_radius: int = 3,
) -> None:
    dataset_root = dataset_root.resolve()
    report: dict[str, object] = {"dataset_root": str(dataset_root), "scenes": []}

    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            scene_stats = eval_charuco_scene(
                scene_dir,
                method=method,
                refine=refine,
                tensor_sigma=tensor_sigma,
                search_radius=search_radius,
            )
            scene_stats["split"] = split
            scene_stats["scene"] = scene_dir.name
            report["scenes"].append(scene_stats)
            print(f"{split}/{scene_dir.name}: {json.dumps(scene_stats, sort_keys=True)}")

    if write_json:
        out = dataset_root / "charuco_detection_report.json"
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {out}")


def eval_charuco_scene(
    scene_dir: Path,
    method: str = "charuco",
    refine: str = "none",
    tensor_sigma: float = 1.5,
    search_radius: int = 3,
) -> dict[str, object]:
    meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
    board = meta.get("board", {})
    if board.get("type") != "charuco":
        raise ValueError(f"{scene_dir} board.type must be charuco for this eval")

    frames = _read_frames(scene_dir / "frames.jsonl")

    gt_path = scene_dir / "gt_charuco_corners.npz"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing {gt_path}")

    gt = np.load(gt_path)
    frame_id = gt["frame_id"].astype(np.int32)
    corner_id = gt["corner_id"].astype(np.int32)
    uvL = gt["uv_left_px"].astype(np.float64)
    uvR = gt["uv_right_px"].astype(np.float64)

    gt_by_frame = _index_gt_by_frame(frame_id, corner_id, uvL, uvR)

    # Build OpenCV board/detector once.
    cv2, aruco, dictionary, charuco_board, detector_params, aruco_detector, charuco_detector = _make_charuco_detector(board)

    # Camera parameters (needed for distortion-aware methods like PnP).
    sim = meta.get("sim_params", {})
    f_um = float(sim.get("f_um", 0.0))
    if f_um <= 0.0:
        f_um = float("nan")

    dist_model = str(sim.get("distortion_model", "none"))
    dist_left = sim.get("distortion_left", {}) if dist_model == "brown" else {}
    dist_right = sim.get("distortion_right", {}) if dist_model == "brown" else {}
    K_left, d_left = _camera_params_from_meta(meta["stereo"]["left"], f_um=f_um, brown=dist_left)
    K_right, d_right = _camera_params_from_meta(meta["stereo"]["right"], f_um=f_um, brown=dist_right)

    per_side_errors: dict[str, list[float]] = {"left": [], "right": []}
    per_side_dx: dict[str, list[float]] = {"left": [], "right": []}
    per_side_dy: dict[str, list[float]] = {"left": [], "right": []}
    per_side_n_detected: dict[str, int] = {"left": 0, "right": 0}
    per_side_n_matched: dict[str, int] = {"left": 0, "right": 0}

    n_frames = 0
    n_frames_with_any_match = 0

    for fr in frames:
        n_frames += 1
        fid = int(fr["frame_id"])
        gt_frame = gt_by_frame.get(fid)
        if gt_frame is None:
            continue

        left_path = scene_dir / "left" / fr["left"]
        right_path = scene_dir / "right" / fr["right"]

        eL, dxL, dyL, n_det_L, n_match_L = _eval_one_image(
            cv2,
            aruco,
            dictionary,
            charuco_board,
            detector_params,
            aruco_detector,
            charuco_detector,
            left_path,
            gt_frame,
            side="left",
            method=method,
            refine=refine,
            tensor_sigma=tensor_sigma,
            search_radius=search_radius,
            camera_matrix=K_left,
            dist_coeffs=d_left,
        )
        eR, dxR, dyR, n_det_R, n_match_R = _eval_one_image(
            cv2,
            aruco,
            dictionary,
            charuco_board,
            detector_params,
            aruco_detector,
            charuco_detector,
            right_path,
            gt_frame,
            side="right",
            method=method,
            refine=refine,
            tensor_sigma=tensor_sigma,
            search_radius=search_radius,
            camera_matrix=K_right,
            dist_coeffs=d_right,
        )

        per_side_errors["left"].extend(eL)
        per_side_errors["right"].extend(eR)
        per_side_dx["left"].extend(dxL)
        per_side_dy["left"].extend(dyL)
        per_side_dx["right"].extend(dxR)
        per_side_dy["right"].extend(dyR)
        per_side_n_detected["left"] += n_det_L
        per_side_n_detected["right"] += n_det_R
        per_side_n_matched["left"] += n_match_L
        per_side_n_matched["right"] += n_match_R

        if n_match_L > 0 or n_match_R > 0:
            n_frames_with_any_match += 1

    stats_left = _summarize(per_side_errors["left"], per_side_dx["left"], per_side_dy["left"])
    stats_right = _summarize(per_side_errors["right"], per_side_dx["right"], per_side_dy["right"])

    return {
        "n_frames": n_frames,
        "n_frames_with_any_match": n_frames_with_any_match,
        "n_gt_rows": int(frame_id.shape[0]),
        "left": {
            "n_detected": per_side_n_detected["left"],
            "n_matched": per_side_n_matched["left"],
            **_stats_to_dict(stats_left),
        },
        "right": {
            "n_detected": per_side_n_detected["right"],
            "n_matched": per_side_n_matched["right"],
            **_stats_to_dict(stats_right),
        },
    }


def collect_charuco_scene_errors(
    scene_dir: Path,
    method: str = "charuco",
    refine: str = "none",
    tensor_sigma: float = 1.5,
    search_radius: int = 3,
) -> dict[str, object]:
    """
    Like `eval_charuco_scene`, but returns raw per-point errors for paper-grade aggregation.

    Methods (2D corner identification):
      - `charuco`: OpenCV ChArUco corners (direct).
      - `homography`: global homography from ArUco marker corners.
      - `tps`: thin-plate spline warp (obj->image) from marker corners.
      - `pnp`: PnP with camera intrinsics + distortion from meta.json (synthetic datasets).
      - `rayfield`: homography + regularized grid residual field (Huber/IRLS).
      - `rayfield_tps`: homography + TPS-smoothed residual field (no robust loss).
      - `rayfield_tps_robust`: homography + TPS-smoothed residual field with robust IRLS (Huber).
      - `mls_affine`, `mls_h`, `pw_affine`, `kfield`, `hybrid`: experimental/ablations.
    """
    meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
    board = meta.get("board", {})
    if board.get("type") != "charuco":
        raise ValueError(f"{scene_dir} board.type must be charuco for this eval")

    frames = _read_frames(scene_dir / "frames.jsonl")

    gt_path = scene_dir / "gt_charuco_corners.npz"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing {gt_path}")

    gt = np.load(gt_path)
    frame_id = gt["frame_id"].astype(np.int32)
    corner_id = gt["corner_id"].astype(np.int32)
    uvL = gt["uv_left_px"].astype(np.float64)
    uvR = gt["uv_right_px"].astype(np.float64)
    gt_by_frame = _index_gt_by_frame(frame_id, corner_id, uvL, uvR)

    cv2, aruco, dictionary, charuco_board, detector_params, aruco_detector, charuco_detector = _make_charuco_detector(board)

    sim = meta.get("sim_params", {})
    f_um = float(sim.get("f_um", 0.0))
    if f_um <= 0.0:
        f_um = float("nan")
    dist_model = str(sim.get("distortion_model", "none"))
    dist_left = sim.get("distortion_left", {}) if dist_model == "brown" else {}
    dist_right = sim.get("distortion_right", {}) if dist_model == "brown" else {}
    K_left, d_left = _camera_params_from_meta(meta["stereo"]["left"], f_um=f_um, brown=dist_left)
    K_right, d_right = _camera_params_from_meta(meta["stereo"]["right"], f_um=f_um, brown=dist_right)

    errors_left: list[float] = []
    errors_right: list[float] = []
    dx_left: list[float] = []
    dy_left: list[float] = []
    dx_right: list[float] = []
    dy_right: list[float] = []
    n_det_left = 0
    n_det_right = 0
    n_match_left = 0
    n_match_right = 0

    for fr in frames:
        fid = int(fr["frame_id"])
        gt_frame = gt_by_frame.get(fid)
        if gt_frame is None:
            continue

        left_path = scene_dir / "left" / fr["left"]
        right_path = scene_dir / "right" / fr["right"]

        eL, dxL, dyL, n_det_L, n_match_L = _eval_one_image(
            cv2,
            aruco,
            dictionary,
            charuco_board,
            detector_params,
            aruco_detector,
            charuco_detector,
            left_path,
            gt_frame,
            side="left",
            method=method,
            refine=refine,
            tensor_sigma=tensor_sigma,
            search_radius=search_radius,
            camera_matrix=K_left,
            dist_coeffs=d_left,
        )
        eR, dxR, dyR, n_det_R, n_match_R = _eval_one_image(
            cv2,
            aruco,
            dictionary,
            charuco_board,
            detector_params,
            aruco_detector,
            charuco_detector,
            right_path,
            gt_frame,
            side="right",
            method=method,
            refine=refine,
            tensor_sigma=tensor_sigma,
            search_radius=search_radius,
            camera_matrix=K_right,
            dist_coeffs=d_right,
        )

        errors_left.extend(eL)
        dx_left.extend(dxL)
        dy_left.extend(dyL)
        errors_right.extend(eR)
        dx_right.extend(dxR)
        dy_right.extend(dyR)
        n_det_left += int(n_det_L)
        n_det_right += int(n_det_R)
        n_match_left += int(n_match_L)
        n_match_right += int(n_match_R)

    return {
        "scene_dir": str(scene_dir),
        "method": str(method),
        "refine": str(refine),
        "left": {"n_detected": n_det_left, "n_matched": n_match_left, "errors": errors_left, "dx": dx_left, "dy": dy_left},
        "right": {
            "n_detected": n_det_right,
            "n_matched": n_match_right,
            "errors": errors_right,
            "dx": dx_right,
            "dy": dy_right,
        },
    }


def _stats_to_dict(stats: ErrorStats | None) -> dict[str, float]:
    if stats is None:
        return {
            "rms_px": float("nan"),
            "mean_px": float("nan"),
            "p50_px": float("nan"),
            "p95_px": float("nan"),
            "max_px": float("nan"),
            "mean_dx_px": float("nan"),
            "mean_dy_px": float("nan"),
            "rms_dx_px": float("nan"),
            "rms_dy_px": float("nan"),
        }
    return {
        "rms_px": stats.rms_px,
        "mean_px": stats.mean_px,
        "p50_px": stats.p50_px,
        "p95_px": stats.p95_px,
        "max_px": stats.max_px,
        "mean_dx_px": stats.mean_dx_px,
        "mean_dy_px": stats.mean_dy_px,
        "rms_dx_px": stats.rms_dx_px,
        "rms_dy_px": stats.rms_dy_px,
    }


def _summarize(errors: list[float], dx: list[float], dy: list[float]) -> ErrorStats | None:
    if not errors:
        return None
    e = np.asarray(errors, dtype=np.float64)
    dxv = np.asarray(dx, dtype=np.float64)
    dyv = np.asarray(dy, dtype=np.float64)
    return ErrorStats(
        n_matched=int(e.size),
        rms_px=float(np.sqrt(np.mean(e**2))),
        mean_px=float(np.mean(e)),
        p50_px=float(np.quantile(e, 0.50)),
        p95_px=float(np.quantile(e, 0.95)),
        max_px=float(np.max(e)),
        mean_dx_px=float(np.mean(dxv)),
        mean_dy_px=float(np.mean(dyv)),
        rms_dx_px=float(np.sqrt(np.mean(dxv**2))),
        rms_dy_px=float(np.sqrt(np.mean(dyv**2))),
    )


def _read_frames(path: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def _index_gt_by_frame(
    frame_id: np.ndarray, corner_id: np.ndarray, uvL: np.ndarray, uvR: np.ndarray
) -> dict[int, dict[str, dict[int, np.ndarray]]]:
    out: dict[int, dict[str, dict[int, np.ndarray]]] = {}
    for fid, cid, l, r in zip(frame_id.tolist(), corner_id.tolist(), uvL, uvR, strict=True):
        f = out.setdefault(int(fid), {"left": {}, "right": {}})
        f["left"][int(cid)] = l
        f["right"][int(cid)] = r
    return out


def _make_charuco_detector(board_meta: dict):
    try:
        import cv2  # type: ignore
        import cv2.aruco as aruco  # type: ignore
    except Exception as e:
        raise RuntimeError("ChArUco eval requires opencv-contrib-python (cv2.aruco).") from e

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
        charuco_board = aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, dictionary)
    elif hasattr(aruco, "CharucoBoard_create"):  # pragma: no cover
        charuco_board = aruco.CharucoBoard_create(squares_x, squares_y, square_size, marker_size, dictionary)
    else:  # pragma: no cover
        raise RuntimeError("cv2.aruco does not expose CharucoBoard APIs in this build.")

    detector_params = aruco.DetectorParameters()
    # Improve marker corner localization (and therefore charuco interpolation) via subpixel refinement.
    # This is a major factor when evaluating compression/blur impacts.
    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 50
    detector_params.cornerRefinementMinAccuracy = 1e-3

    # OpenCV >= 4.7 uses CharucoDetector; older builds expose interpolateCornersCharuco.
    charuco_detector = None
    if hasattr(aruco, "CharucoDetector"):
        charuco_detector = aruco.CharucoDetector(charuco_board)
        if hasattr(charuco_detector, "setDetectorParameters"):
            charuco_detector.setDetectorParameters(detector_params)

    aruco_detector = None
    if charuco_detector is None and hasattr(aruco, "ArucoDetector"):
        aruco_detector = aruco.ArucoDetector(dictionary, detector_params)

    return cv2, aruco, dictionary, charuco_board, detector_params, aruco_detector, charuco_detector


def _camera_params_from_meta(view_meta: dict, f_um: float, brown: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Build OpenCV-style (K, dist) from dataset meta conventions.

    This repo's pixel convention matches `stereocomplex.core.geometry.sensor_um_to_pixel`:
    pixel centers are at integer coordinates, and the principal point is at the crop center.
    """
    f_um = float(f_um)
    if not np.isfinite(f_um) or f_um <= 0.0:
        # Keep shapes valid; methods that require a real camera should error out elsewhere.
        return np.eye(3, dtype=np.float64), np.zeros((5,), dtype=np.float64)

    sensor = view_meta.get("sensor", {})
    preprocess = view_meta.get("preprocess", {})

    pitch_um = float(sensor.get("pixel_pitch_um", 1.0))
    bin_x, bin_y = sensor.get("binning_xy", [1, 1])
    pitch_x_um = pitch_um * float(bin_x)
    pitch_y_um = pitch_um * float(bin_y)

    resize_x, resize_y = preprocess.get("resize_xy", [1.0, 1.0])
    crop_x, crop_y, crop_w, crop_h = preprocess.get("crop_xywh_px", [0, 0, 0, 0])
    crop_w = float(crop_w)
    crop_h = float(crop_h)
    resize_x = float(resize_x)
    resize_y = float(resize_y)

    fx = f_um * resize_x / pitch_x_um
    fy = f_um * resize_y / pitch_y_um
    cx = crop_w * 0.5 * resize_x - 0.5
    cy = crop_h * 0.5 * resize_y - 0.5

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(
        [
            float(brown.get("k1", 0.0)),
            float(brown.get("k2", 0.0)),
            float(brown.get("p1", 0.0)),
            float(brown.get("p2", 0.0)),
            float(brown.get("k3", 0.0)),
        ],
        dtype=np.float64,
    )
    return K, dist


def _eval_one_image(
    cv2,
    aruco,
    dictionary,
    charuco_board,
    detector_params,
    aruco_detector,
    charuco_detector,
    image_path: Path,
    gt_frame: dict[str, dict[int, np.ndarray]],
    side: str,
    method: str,
    refine: str,
    tensor_sigma: float,
    search_radius: int,
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray | None,
) -> tuple[list[float], list[float], list[float], int, int]:
    from stereocomplex.core.image_io import load_gray_u8

    img = load_gray_u8(image_path)
    if img.size == 0:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    method = str(method)
    refine = str(refine)

    # Prefer CharucoDetector if available.
    if charuco_detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)
        if method in (
            "homography",
            "pnp",
            "mls",
            "mls_affine",
            "mls_h",
            "pw_affine",
            "tps",
            "kfield",
            "rayfield",
            "rayfield_tps",
            "rayfield_tps_robust",
        ):
            if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
                return [], [], [], 0, 0
        else:
            if charuco_ids is None or charuco_corners is None:
                return [], [], [], 0, 0
    else:
        if aruco_detector is not None:
            corners, ids, _rejected = aruco_detector.detectMarkers(img)
        else:  # pragma: no cover
            corners, ids, _rejected = aruco.detectMarkers(img, dictionary, parameters=detector_params)

        if ids is None or len(ids) == 0:
            return [], [], [], 0, 0

        marker_corners, marker_ids = corners, ids
        charuco_corners, charuco_ids = None, None
        if method in ("charuco", "hybrid"):
            if not hasattr(aruco, "interpolateCornersCharuco"):
                raise RuntimeError("OpenCV build has no CharucoDetector or interpolateCornersCharuco.")

            ret = aruco.interpolateCornersCharuco(corners, ids, img, charuco_board)
            if ret is None:
                return [], [], [], int(len(ids)), 0
            charuco_corners, charuco_ids, _ = ret
            if charuco_ids is None or charuco_corners is None:
                return [], [], [], int(len(ids)), 0

    if method == "mls":
        # Backward-compatible alias.
        method = "mls_affine"

    if method == "hybrid":
        # Base prediction from markers (local projective), then correct with a locally
        # interpolated residual field learned from detected ChArUco corners.
        if charuco_ids is None or charuco_corners is None:
            return [], [], [], 0, 0
        if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
            return [], [], [], 0, 0

        det_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        det_xy = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
        # Match dataset pixel-center convention.
        det_xy = det_xy - 0.5

        marker_ids_arr = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
        board_ids = np.asarray(charuco_board.getIds(), dtype=np.int32).reshape(-1)
        board_obj = charuco_board.getObjPoints()
        id_to_obj2 = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

        obj_pts = []
        img_pts = []
        for mid, mc in zip(marker_ids_arr.tolist(), marker_corners, strict=True):
            o = id_to_obj2.get(int(mid))
            if o is None:
                continue
            mc = np.asarray(mc, dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] != 4 or o.shape[0] != 4:
                continue
            obj_pts.append(o)
            img_pts.append(mc)
        if not obj_pts:
            return [], [], [], 0, 0
        obj_pts = np.concatenate(obj_pts, axis=0)
        img_pts = np.concatenate(img_pts, axis=0)

        chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)[:, :2]
        base_xy = _predict_points_mls_homography(obj_pts, img_pts, chess)

        if det_ids.size < 12:
            charuco_ids = np.arange(chess.shape[0], dtype=np.int32)
            charuco_xy = base_xy
        else:
            res = det_xy - base_xy[det_ids]
            # Residual field is fit in board coordinates using detected ChArUco corners.
            res_pred = _predict_points_mls_affine(chess[det_ids], res, chess, k=min(80, det_ids.size))
            charuco_ids = np.arange(chess.shape[0], dtype=np.int32)
            charuco_xy = base_xy + res_pred

    elif method == "kfield":
        if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
            return [], [], [], 0, 0

        marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
        board_ids = np.asarray(charuco_board.getIds(), dtype=np.int32).reshape(-1)
        board_obj = charuco_board.getObjPoints()
        id_to_obj2 = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

        obj_pts = []
        img_pts = []
        for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
            o = id_to_obj2.get(int(mid))
            if o is None:
                continue
            mc = np.asarray(mc, dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] != 4 or o.shape[0] != 4:
                continue
            obj_pts.append(o)
            img_pts.append(mc)
        if not obj_pts:
            return [], [], [], 0, 0
        obj_pts = np.concatenate(obj_pts, axis=0)
        img_pts = np.concatenate(img_pts, axis=0)

        chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)[:, :2]
        charuco_ids = np.arange(chess.shape[0], dtype=np.int32)
        charuco_xy = _predict_points_affine_field(obj_pts, img_pts, chess)

    elif method == "rayfield":
        if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
            return [], [], [], 0, 0

        marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
        board_ids = np.asarray(charuco_board.getIds(), dtype=np.int32).reshape(-1)
        board_obj = charuco_board.getObjPoints()
        id_to_obj2 = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

        obj_pts = []
        img_pts = []
        for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
            o = id_to_obj2.get(int(mid))
            if o is None:
                continue
            mc = np.asarray(mc, dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] != 4 or o.shape[0] != 4:
                continue
            obj_pts.append(o)
            img_pts.append(mc)
        if not obj_pts:
            return [], [], [], 0, 0
        obj_pts = np.concatenate(obj_pts, axis=0)
        img_pts = np.concatenate(img_pts, axis=0)

        chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)[:, :2]
        charuco_ids = np.arange(chess.shape[0], dtype=np.int32)
        charuco_xy = _predict_points_rayfield(obj_pts, img_pts, chess)

    elif method == "rayfield_tps":
        if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
            return [], [], [], 0, 0

        marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
        board_ids = np.asarray(charuco_board.getIds(), dtype=np.int32).reshape(-1)
        board_obj = charuco_board.getObjPoints()
        id_to_obj2 = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

        obj_pts = []
        img_pts = []
        for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
            o = id_to_obj2.get(int(mid))
            if o is None:
                continue
            mc = np.asarray(mc, dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] != 4 or o.shape[0] != 4:
                continue
            obj_pts.append(o)
            img_pts.append(mc)
        if not obj_pts:
            return [], [], [], 0, 0
        obj_pts = np.concatenate(obj_pts, axis=0)
        img_pts = np.concatenate(img_pts, axis=0)

        chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)[:, :2]
        charuco_ids = np.arange(chess.shape[0], dtype=np.int32)
        charuco_xy = _predict_points_rayfield_tps(obj_pts, img_pts, chess)

    elif method == "rayfield_tps_robust":
        if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
            return [], [], [], 0, 0

        marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
        board_ids = np.asarray(charuco_board.getIds(), dtype=np.int32).reshape(-1)
        board_obj = charuco_board.getObjPoints()
        id_to_obj2 = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

        obj_pts = []
        img_pts = []
        for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
            o = id_to_obj2.get(int(mid))
            if o is None:
                continue
            mc = np.asarray(mc, dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] != 4 or o.shape[0] != 4:
                continue
            obj_pts.append(o)
            img_pts.append(mc)
        if not obj_pts:
            return [], [], [], 0, 0
        obj_pts = np.concatenate(obj_pts, axis=0)
        img_pts = np.concatenate(img_pts, axis=0)

        chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)[:, :2]
        charuco_ids = np.arange(chess.shape[0], dtype=np.int32)
        charuco_xy = _predict_points_rayfield_tps_robust(obj_pts, img_pts, chess)

    elif method == "mls_affine":
        marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
        board_ids = np.asarray(charuco_board.getIds(), dtype=np.int32).reshape(-1)
        board_obj = charuco_board.getObjPoints()
        id_to_obj2 = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

        obj_pts = []
        img_pts = []
        for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
            o = id_to_obj2.get(int(mid))
            if o is None:
                continue
            mc = np.asarray(mc, dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] != 4 or o.shape[0] != 4:
                continue
            obj_pts.append(o)
            img_pts.append(mc)
        if not obj_pts:
            return [], [], [], 0, 0
        obj_pts = np.concatenate(obj_pts, axis=0)  # (N,2) in board units (mm)
        img_pts = np.concatenate(img_pts, axis=0)  # (N,2) in px

        chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)[:, :2]
        charuco_ids = np.arange(chess.shape[0], dtype=np.int32)
        charuco_xy = _predict_points_mls_affine(obj_pts, img_pts, chess)
    elif method == "mls_h":
        marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
        board_ids = np.asarray(charuco_board.getIds(), dtype=np.int32).reshape(-1)
        board_obj = charuco_board.getObjPoints()
        id_to_obj2 = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

        obj_pts = []
        img_pts = []
        for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
            o = id_to_obj2.get(int(mid))
            if o is None:
                continue
            mc = np.asarray(mc, dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] != 4 or o.shape[0] != 4:
                continue
            obj_pts.append(o)
            img_pts.append(mc)
        if not obj_pts:
            return [], [], [], 0, 0
        obj_pts = np.concatenate(obj_pts, axis=0)
        img_pts = np.concatenate(img_pts, axis=0)

        chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)[:, :2]
        charuco_ids = np.arange(chess.shape[0], dtype=np.int32)
        charuco_xy = _predict_points_mls_homography(obj_pts, img_pts, chess)
    elif method == "pw_affine":
        marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
        board_ids = np.asarray(charuco_board.getIds(), dtype=np.int32).reshape(-1)
        board_obj = charuco_board.getObjPoints()
        id_to_obj2 = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

        obj_pts = []
        img_pts = []
        for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
            o = id_to_obj2.get(int(mid))
            if o is None:
                continue
            mc = np.asarray(mc, dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] != 4 or o.shape[0] != 4:
                continue
            obj_pts.append(o)
            img_pts.append(mc)
        if not obj_pts:
            return [], [], [], 0, 0
        obj_pts = np.concatenate(obj_pts, axis=0)
        img_pts = np.concatenate(img_pts, axis=0)

        chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)[:, :2]
        charuco_ids = np.arange(chess.shape[0], dtype=np.int32)
        charuco_xy = _predict_points_piecewise_affine(obj_pts, img_pts, chess)
    elif method == "tps":
        marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
        board_ids = np.asarray(charuco_board.getIds(), dtype=np.int32).reshape(-1)
        board_obj = charuco_board.getObjPoints()
        id_to_obj2 = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

        obj_pts = []
        img_pts = []
        for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
            o = id_to_obj2.get(int(mid))
            if o is None:
                continue
            mc = np.asarray(mc, dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] != 4 or o.shape[0] != 4:
                continue
            obj_pts.append(o)
            img_pts.append(mc)
        if not obj_pts:
            return [], [], [], 0, 0
        obj_pts = np.concatenate(obj_pts, axis=0)
        img_pts = np.concatenate(img_pts, axis=0)

        chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)[:, :2]
        charuco_ids = np.arange(chess.shape[0], dtype=np.int32)
        charuco_xy = _predict_points_tps(obj_pts, img_pts, chess)
    elif method == "pnp":
        marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
        board_ids = np.asarray(charuco_board.getIds(), dtype=np.int32).reshape(-1)
        board_obj = charuco_board.getObjPoints()
        id_to_obj = {int(i): np.asarray(p, dtype=np.float64) for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

        obj_pts = []
        img_pts = []
        for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
            o = id_to_obj.get(int(mid))
            if o is None:
                continue
            mc = np.asarray(mc, dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] != 4 or o.shape[0] != 4:
                continue
            obj_pts.append(o)
            img_pts.append(mc)
        if not obj_pts:
            return [], [], [], 0, 0
        obj_pts = np.concatenate(obj_pts, axis=0)
        img_pts = np.concatenate(img_pts, axis=0)

        if camera_matrix is None or dist_coeffs is None:
            raise RuntimeError("pnp method requires camera_matrix and dist_coeffs from meta.json")

        ok, rvec, tvec, _inliers = cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0,
            iterationsCount=200,
            confidence=0.999,
        )
        if not ok:
            return [], [], [], 0, 0

        chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)
        charuco_ids = np.arange(chess.shape[0], dtype=np.int32)
        proj, _jac = cv2.projectPoints(chess, rvec, tvec, camera_matrix, dist_coeffs)
        charuco_xy = proj.reshape(-1, 2).astype(np.float64)
    elif method == "homography":
        if charuco_detector is None:
            raise RuntimeError("homography method requires CharucoDetector (OpenCV >= 4.7).")

        marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
        board_ids = np.asarray(charuco_board.getIds(), dtype=np.int32).reshape(-1)
        board_obj = charuco_board.getObjPoints()
        id_to_obj = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

        obj_pts = []
        img_pts = []
        for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
            o = id_to_obj.get(int(mid))
            if o is None:
                continue
            mc = np.asarray(mc, dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] != 4:
                continue
            obj_pts.append(o)
            img_pts.append(mc)
        if not obj_pts:
            return [], [], [], 0, 0
        obj_pts = np.concatenate(obj_pts, axis=0)
        img_pts = np.concatenate(img_pts, axis=0)

        H, _mask = cv2.findHomography(obj_pts, img_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None:
            return [], [], [], 0, 0

        chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)[:, :2]
        charuco_ids = np.arange(chess.shape[0], dtype=np.int32)
        charuco_xy = cv2.perspectiveTransform(chess.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2).astype(np.float64)
    elif method == "charuco":
        charuco_ids = charuco_ids.reshape(-1).astype(np.int32)
        charuco_xy = charuco_corners.reshape(-1, 2).astype(np.float64)
    else:
        raise ValueError(
            "method must be charuco|homography|pnp|mls|mls_h|pw_affine|tps|hybrid|kfield|rayfield|rayfield_tps|rayfield_tps_robust"
        )

    # Coordinate convention:
    # - OpenCV ChArUco corners are typically reported in a (i+0.5, j+0.5) convention.
    # - For the homography path (built from marker corners + perspectiveTransform), the result is already
    #   consistent with the dataset's internal pixel-center convention.
    if method == "charuco":
        charuco_xy = charuco_xy - 0.5

    if refine == "tensor":
        charuco_xy = _refine_points_tensor_symmetry(
            cv2,
            img,
            charuco_xy,
            search_radius=float(search_radius),
            tensor_sigma=float(tensor_sigma),
        )
    elif refine == "lines":
        charuco_xy = _refine_points_tensor_lines(
            cv2,
            img,
            charuco_xy,
            search_radius=float(search_radius),
            tensor_sigma=float(tensor_sigma),
        )
    elif refine == "lsq":
        charuco_xy = _refine_points_tensor_lsq(
            cv2,
            img,
            charuco_xy,
            search_radius=float(search_radius),
            tensor_sigma=float(tensor_sigma),
        )
    elif refine == "noble":
        charuco_xy = _refine_points_tensor_noble(
            cv2,
            img,
            charuco_xy,
            search_radius=float(search_radius),
            tensor_sigma=float(tensor_sigma),
        )
    elif refine != "none":
        raise ValueError("refine must be none|tensor|lines|lsq|noble")

    gt_map = gt_frame[side]
    errors: list[float] = []
    dx_list: list[float] = []
    dy_list: list[float] = []
    n_matched = 0
    for cid, xy in zip(charuco_ids.tolist(), charuco_xy, strict=True):
        gt_xy = gt_map.get(int(cid))
        if gt_xy is None:
            continue
        diff = xy - gt_xy
        dx = float(diff[0])
        dy = float(diff[1])
        errors.append(float(np.linalg.norm(diff)))
        dx_list.append(dx)
        dy_list.append(dy)
        n_matched += 1

    return errors, dx_list, dy_list, int(charuco_ids.size), n_matched


def _refine_points_tensor_symmetry(
    cv2,
    img_u8: np.ndarray,
    pts_xy: np.ndarray,
    search_radius: float,
    tensor_sigma: float,
) -> np.ndarray:
    """
    Pycaso-like second pass: estimate local axes from the structure tensor, then maximize a
    quadrant-symmetry score evaluated along those axes.
    """
    search_radius = float(max(0.25, search_radius))
    tensor_sigma = float(max(0.8, tensor_sigma))

    img = img_u8.astype(np.float32) / 255.0
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)

    H, W = img_u8.shape[:2]
    out = pts_xy.astype(np.float64, copy=True)

    # Window for tensor estimation.
    win_r = int(max(3, round(2.5 * tensor_sigma)))
    ys, xs = np.mgrid[-win_r : win_r + 1, -win_r : win_r + 1]
    w = np.exp(-(xs * xs + ys * ys) / (2.0 * tensor_sigma * tensor_sigma)).astype(np.float32)

    # Candidate offsets (subpixel grid).
    step = 0.25
    offs = np.arange(-search_radius, search_radius + 1e-9, step, dtype=np.float64)

    # Radii for quadrant sampling along axes.
    radii = np.array([1.5, 2.5, 3.5], dtype=np.float64)

    for i in range(out.shape[0]):
        x0, y0 = out[i]
        xi = int(np.rint(x0))
        yi = int(np.rint(y0))
        if xi < win_r + 4 or xi >= (W - win_r - 4) or yi < win_r + 4 or yi >= (H - win_r - 4):
            continue

        gx = Ix[yi - win_r : yi + win_r + 1, xi - win_r : xi + win_r + 1]
        gy = Iy[yi - win_r : yi + win_r + 1, xi - win_r : xi + win_r + 1]

        Sxx = float(np.sum(w * gx * gx))
        Sxy = float(np.sum(w * gx * gy))
        Syy = float(np.sum(w * gy * gy))
        J = np.array([[Sxx, Sxy], [Sxy, Syy]], dtype=np.float64)
        vals, vecs = np.linalg.eigh(J)
        if float(vals[1]) < 1e-10:
            continue
        # vecs columns are eigenvectors; use them as orthonormal axes.
        e1 = vecs[:, 1]
        e2 = vecs[:, 0]

        best_score = -1.0
        best_xy = (x0, y0)

        for dy in offs:
            for dx in offs:
                x = x0 + dx
                y = y0 + dy
                s = 0.0
                for rr in radii:
                    ppp = _bilinear(img, x + rr * (e1[0] + e2[0]), y + rr * (e1[1] + e2[1]))
                    pmm = _bilinear(img, x - rr * (e1[0] + e2[0]), y - rr * (e1[1] + e2[1]))
                    ppm = _bilinear(img, x + rr * (e1[0] - e2[0]), y + rr * (e1[1] - e2[1]))
                    pmp = _bilinear(img, x - rr * (e1[0] - e2[0]), y - rr * (e1[1] - e2[1]))
                    s += abs((ppp + pmm) - (ppm + pmp))
                if s > best_score:
                    best_score = s
                    best_xy = (x, y)

        out[i, 0] = best_xy[0]
        out[i, 1] = best_xy[1]

    return out


def _refine_points_tensor_noble(
    cv2,
    img_u8: np.ndarray,
    pts_xy: np.ndarray,
    search_radius: float,
    tensor_sigma: float,
) -> np.ndarray:
    """
    Second pass based on a rotation-invariant structure tensor cornerness measure.

    Uses the "Noble" measure: det(J) / trace(J), where J is the (Gaussian-smoothed)
    structure tensor built from image gradients. For each point, searches a small
    window around the initial estimate and refines to subpixel precision via a
    1D parabolic fit around the maximum.
    """
    search_radius = float(max(1.0, search_radius))
    tensor_sigma = float(max(0.8, tensor_sigma))

    img = img_u8.astype(np.float32) / 255.0
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)

    # Smoothed gradient products (structure tensor components).
    A = cv2.GaussianBlur(Ix * Ix, ksize=(0, 0), sigmaX=tensor_sigma, sigmaY=tensor_sigma, borderType=cv2.BORDER_REFLECT)
    B = cv2.GaussianBlur(Ix * Iy, ksize=(0, 0), sigmaX=tensor_sigma, sigmaY=tensor_sigma, borderType=cv2.BORDER_REFLECT)
    C = cv2.GaussianBlur(Iy * Iy, ksize=(0, 0), sigmaX=tensor_sigma, sigmaY=tensor_sigma, borderType=cv2.BORDER_REFLECT)

    trace = A + C
    det = A * C - B * B
    R = det / (trace + 1e-12)

    H, W = img_u8.shape[:2]
    out = pts_xy.astype(np.float64, copy=True)
    r = int(round(search_radius))
    ys, xs = np.mgrid[-r : r + 1, -r : r + 1]
    # Favor the peak closest to the initial estimate to avoid snapping to nearby
    # high-contrast features (e.g., marker micro-corners).
    w = np.exp(-(xs * xs + ys * ys) / (2.0 * max(1.0, 0.5 * r) ** 2)).astype(np.float32)

    def _parabolic_delta(v_m1: float, v_0: float, v_p1: float) -> float:
        denom = v_m1 - 2.0 * v_0 + v_p1
        if abs(denom) < 1e-12:
            return 0.0
        d = 0.5 * (v_m1 - v_p1) / denom
        # Keep refinement local and stable.
        return float(np.clip(d, -1.0, 1.0))

    for i in range(out.shape[0]):
        x0, y0 = out[i]
        xi = int(np.rint(x0))
        yi = int(np.rint(y0))
        if xi < r + 1 or xi >= (W - r - 1) or yi < r + 1 or yi >= (H - r - 1):
            continue

        patch = R[yi - r : yi + r + 1, xi - r : xi + r + 1]
        if patch.size == 0:
            continue

        flat_idx = int(np.argmax(patch * w))
        my, mx = np.unravel_index(flat_idx, patch.shape)
        px = (xi - r) + int(mx)
        py = (yi - r) + int(my)

        if px < 1 or px >= (W - 1) or py < 1 or py >= (H - 1):
            continue

        v0 = float(R[py, px])
        if not np.isfinite(v0) or v0 <= 0.0:
            continue

        dx = _parabolic_delta(float(R[py, px - 1]), v0, float(R[py, px + 1]))
        dy = _parabolic_delta(float(R[py - 1, px]), v0, float(R[py + 1, px]))

        out[i, 0] = float(px) + dx
        out[i, 1] = float(py) + dy

    return out


def _refine_points_tensor_lines(
    cv2,
    img_u8: np.ndarray,
    pts_xy: np.ndarray,
    search_radius: float,
    tensor_sigma: float,
) -> np.ndarray:
    """
    Pycaso-like second pass using the structure tensor to estimate two edge normals,
    then re-localizing each edge by 1D search along its normal and intersecting the
    two edge lines.

    This targets a more "geometric" corner center than simply maximizing a cornerness map.
    """
    search_radius = float(max(0.5, search_radius))
    tensor_sigma = float(max(0.8, tensor_sigma))

    img = img_u8.astype(np.float32) / 255.0
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)

    H, W = img_u8.shape[:2]
    out = pts_xy.astype(np.float64, copy=True)

    win_r = int(max(3, round(2.5 * tensor_sigma)))
    ys, xs = np.mgrid[-win_r : win_r + 1, -win_r : win_r + 1]
    w = np.exp(-(xs * xs + ys * ys) / (2.0 * tensor_sigma * tensor_sigma)).astype(np.float32)

    # 1D search settings.
    step = 0.25
    offs = np.arange(-search_radius, search_radius + 1e-9, step, dtype=np.float64)
    line_half_len = int(max(2, round(2.0 * tensor_sigma)))
    line_s = np.arange(-line_half_len, line_half_len + 1, 1.0, dtype=np.float64)

    def _edge_score_at(p: np.ndarray, n: np.ndarray, t: np.ndarray) -> float:
        s = 0.0
        for ss in line_s:
            x = float(p[0] + ss * t[0])
            y = float(p[1] + ss * t[1])
            gx = _bilinear(Ix, x, y)
            gy = _bilinear(Iy, x, y)
            s += abs(gx * float(n[0]) + gy * float(n[1]))
        return s

    def _refine_edge_offset(p0: np.ndarray, n: np.ndarray) -> float:
        # Edge tangent is perpendicular to its normal.
        t = np.array([-n[1], n[0]], dtype=np.float64)

        scores = np.empty((offs.size,), dtype=np.float64)
        for k, a in enumerate(offs.tolist()):
            scores[k] = _edge_score_at(p0 + a * n, n, t)

        k0 = int(np.argmax(scores))
        if 0 < k0 < (scores.size - 1):
            v_m1 = scores[k0 - 1]
            v_0 = scores[k0]
            v_p1 = scores[k0 + 1]
            denom = v_m1 - 2.0 * v_0 + v_p1
            if abs(denom) > 1e-12:
                d = 0.5 * (v_m1 - v_p1) / denom
                d = float(np.clip(d, -1.0, 1.0))
                return float(offs[k0] + d * step)
        return float(offs[k0])

    for i in range(out.shape[0]):
        x0, y0 = out[i]
        xi = int(np.rint(x0))
        yi = int(np.rint(y0))
        if xi < win_r + 3 or xi >= (W - win_r - 3) or yi < win_r + 3 or yi >= (H - win_r - 3):
            continue

        gx = Ix[yi - win_r : yi + win_r + 1, xi - win_r : xi + win_r + 1]
        gy = Iy[yi - win_r : yi + win_r + 1, xi - win_r : xi + win_r + 1]

        Sxx = float(np.sum(w * gx * gx))
        Sxy = float(np.sum(w * gx * gy))
        Syy = float(np.sum(w * gy * gy))
        J = np.array([[Sxx, Sxy], [Sxy, Syy]], dtype=np.float64)
        vals, vecs = np.linalg.eigh(J)
        if float(vals[0]) < 1e-10 or float(vals[1]) < 1e-10:
            continue

        # Eigenvectors of J are the principal directions of gradient energy (edge normals).
        n1 = vecs[:, 1].astype(np.float64, copy=False)
        n2 = vecs[:, 0].astype(np.float64, copy=False)

        p0 = np.array([x0, y0], dtype=np.float64)
        a1 = _refine_edge_offset(p0, n1)
        a2 = _refine_edge_offset(p0, n2)

        p1 = p0 + a1 * n1
        p2 = p0 + a2 * n2

        c1 = float(n1[0] * p1[0] + n1[1] * p1[1])
        c2 = float(n2[0] * p2[0] + n2[1] * p2[1])

        M = np.array([[n1[0], n1[1]], [n2[0], n2[1]]], dtype=np.float64)
        detM = float(np.linalg.det(M))
        if abs(detM) < 1e-8:
            continue
        xy = np.linalg.solve(M, np.array([c1, c2], dtype=np.float64))

        if float(np.linalg.norm(xy - p0)) > (search_radius + 1.0):
            continue

        out[i, 0] = float(xy[0])
        out[i, 1] = float(xy[1])

    return out


def _refine_points_tensor_lsq(
    cv2,
    img_u8: np.ndarray,
    pts_xy: np.ndarray,
    search_radius: float,
    tensor_sigma: float,
) -> np.ndarray:
    """
    Structure-tensor based corner refinement via least-squares intersection of local
    gradient-normals (line constraints).

    For pixels p in a window around the initial estimate, treat the local edge normal
    as n = grad / ||grad|| and add a constraint nᵀ x = nᵀ p. Solve the weighted normal
    equations for x (the corner).
    """
    search_radius = float(max(0.5, search_radius))
    tensor_sigma = float(max(0.8, tensor_sigma))

    img = img_u8.astype(np.float32) / 255.0
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)

    H, W = img_u8.shape[:2]
    out = pts_xy.astype(np.float64, copy=True)

    win_r = int(max(3, round(2.5 * tensor_sigma)))
    ys, xs = np.mgrid[-win_r : win_r + 1, -win_r : win_r + 1]
    w = np.exp(-(xs * xs + ys * ys) / (2.0 * tensor_sigma * tensor_sigma)).astype(np.float32)
    mag_thresh = 1e-3

    for i in range(out.shape[0]):
        x0, y0 = out[i]
        xi = int(np.rint(x0))
        yi = int(np.rint(y0))
        if xi < win_r + 2 or xi >= (W - win_r - 2) or yi < win_r + 2 or yi >= (H - win_r - 2):
            continue

        gx = Ix[yi - win_r : yi + win_r + 1, xi - win_r : xi + win_r + 1].astype(np.float64, copy=False)
        gy = Iy[yi - win_r : yi + win_r + 1, xi - win_r : xi + win_r + 1].astype(np.float64, copy=False)

        mag = np.sqrt(gx * gx + gy * gy)
        m = mag > mag_thresh
        if int(np.count_nonzero(m)) < 10:
            continue

        nx = np.zeros_like(gx)
        ny = np.zeros_like(gy)
        nx[m] = gx[m] / mag[m]
        ny[m] = gy[m] / mag[m]

        # Pixel-center coordinates in the global image frame.
        px = (xi + xs).astype(np.float64)
        py = (yi + ys).astype(np.float64)

        # Weights: Gaussian window * gradient magnitude (favor strong edges).
        ww = (w.astype(np.float64) * mag)[m]
        nxv = nx[m]
        nyv = ny[m]
        dot = nxv * px[m] + nyv * py[m]

        A11 = float(np.sum(ww * nxv * nxv))
        A12 = float(np.sum(ww * nxv * nyv))
        A22 = float(np.sum(ww * nyv * nyv))
        b1 = float(np.sum(ww * nxv * dot))
        b2 = float(np.sum(ww * nyv * dot))

        detA = A11 * A22 - A12 * A12
        if abs(detA) < 1e-10:
            continue

        x = (A22 * b1 - A12 * b2) / detA
        y = (-A12 * b1 + A11 * b2) / detA
        xy = np.array([x, y], dtype=np.float64)

        if float(np.linalg.norm(xy - np.array([x0, y0], dtype=np.float64))) > (search_radius + 1.0):
            continue

        out[i, 0] = float(xy[0])
        out[i, 1] = float(xy[1])

    return out


def _predict_points_mls_affine(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    k: int = 40,
    sigma_obj: float | None = None,
) -> np.ndarray:
    """
    Moving least squares (affine) mapping from board coords -> image coords.

    For each query point, fits an affine map using the k strongest Gaussian-weighted
    correspondences in object space.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    N = int(obj_xy.shape[0])
    k = int(max(6, min(k, N)))
    if sigma_obj is None:
        # Heuristic: use a few square sizes worth of influence, derived from marker spacing.
        # If correspondences are very sparse, inflate sigma.
        span = float(np.median(np.sqrt(np.sum((obj_xy - np.median(obj_xy, axis=0)) ** 2, axis=1))) + 1e-9)
        sigma_obj = max(10.0, 0.25 * span)
    sigma2 = float(sigma_obj) ** 2

    out = np.empty((query_xy.shape[0], 2), dtype=np.float64)

    for i in range(query_xy.shape[0]):
        q = query_xy[i]
        d2 = np.sum((obj_xy - q[None, :]) ** 2, axis=1)
        w = np.exp(-0.5 * d2 / sigma2)

        # Take k best weights.
        if N > k:
            idx = np.argpartition(w, -k)[-k:]
        else:
            idx = np.arange(N)
        ww = w[idx]
        if float(np.max(ww)) <= 1e-12:
            out[i] = np.array([np.nan, np.nan], dtype=np.float64)
            continue

        X = np.concatenate([obj_xy[idx], np.ones((idx.shape[0], 1), dtype=np.float64)], axis=1)  # (k,3)
        u = img_uv[idx, 0]
        v = img_uv[idx, 1]

        # Weighted normal equations: (X^T W X) a = X^T W u
        WX = X * ww[:, None]
        A = X.T @ WX  # (3,3)
        bu = X.T @ (ww * u)
        bv = X.T @ (ww * v)
        detA = float(np.linalg.det(A))
        if abs(detA) < 1e-12:
            out[i] = np.array([np.nan, np.nan], dtype=np.float64)
            continue
        au = np.linalg.solve(A, bu)
        av = np.linalg.solve(A, bv)
        qh = np.array([q[0], q[1], 1.0], dtype=np.float64)
        out[i, 0] = float(qh @ au)
        out[i, 1] = float(qh @ av)

    return out


def _predict_points_affine_field(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    grid_size: tuple[int, int] = (9, 6),
    k: int = 80,
    sigma_obj: float | None = None,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """
    Smooth "field of local intrinsics" surrogate.

    Instead of a single global K, estimate a low-frequency field of local affine maps
    from board (x,y) to image (u,v), by fitting affines at anchor locations and
    smoothing/interpolating their parameters.
    """
    import cv2  # type: ignore

    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    N = int(obj_xy.shape[0])
    if N < 6:
        return _predict_points_mls_affine(obj_xy, img_uv, query_xy)

    nx, ny = (int(grid_size[0]), int(grid_size[1]))
    nx = max(2, nx)
    ny = max(2, ny)

    xmin = float(np.min(obj_xy[:, 0]))
    xmax = float(np.max(obj_xy[:, 0]))
    ymin = float(np.min(obj_xy[:, 1]))
    ymax = float(np.max(obj_xy[:, 1]))
    if not np.isfinite([xmin, xmax, ymin, ymax]).all() or (xmax - xmin) < 1e-9 or (ymax - ymin) < 1e-9:
        return _predict_points_mls_affine(obj_xy, img_uv, query_xy)

    if sigma_obj is None:
        span = float(np.median(np.sqrt(np.sum((obj_xy - np.median(obj_xy, axis=0)) ** 2, axis=1))) + 1e-9)
        sigma_obj = max(10.0, 0.35 * span)
    sigma2 = float(sigma_obj) ** 2

    # Global affine fallback.
    Xg = np.concatenate([obj_xy, np.ones((N, 1), dtype=np.float64)], axis=1)
    Ag = Xg.T @ Xg
    bg_u = Xg.T @ img_uv[:, 0]
    bg_v = Xg.T @ img_uv[:, 1]
    try:
        au_g = np.linalg.solve(Ag, bg_u)
        av_g = np.linalg.solve(Ag, bg_v)
    except np.linalg.LinAlgError:
        return _predict_points_mls_affine(obj_xy, img_uv, query_xy)

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)

    Pu = np.empty((ny, nx, 3), dtype=np.float64)
    Pv = np.empty((ny, nx, 3), dtype=np.float64)

    k = int(max(6, min(int(k), N)))
    for j in range(ny):
        for i in range(nx):
            q = np.array([xs[i], ys[j]], dtype=np.float64)
            d2 = np.sum((obj_xy - q[None, :]) ** 2, axis=1)
            w = np.exp(-0.5 * d2 / sigma2)
            if N > k:
                idx = np.argpartition(w, -k)[-k:]
            else:
                idx = np.arange(N)
            ww = w[idx]
            if float(np.max(ww)) <= 1e-12:
                Pu[j, i] = au_g
                Pv[j, i] = av_g
                continue

            X = np.concatenate([obj_xy[idx], np.ones((idx.shape[0], 1), dtype=np.float64)], axis=1)
            u = img_uv[idx, 0]
            v = img_uv[idx, 1]
            WX = X * ww[:, None]
            A = X.T @ WX
            bu = X.T @ (ww * u)
            bv = X.T @ (ww * v)
            try:
                Pu[j, i] = np.linalg.solve(A, bu)
                Pv[j, i] = np.linalg.solve(A, bv)
            except np.linalg.LinAlgError:
                Pu[j, i] = au_g
                Pv[j, i] = av_g

    for c in range(3):
        Pu[:, :, c] = cv2.GaussianBlur(Pu[:, :, c].astype(np.float32), ksize=(0, 0), sigmaX=float(smooth_sigma)).astype(
            np.float64
        )
        Pv[:, :, c] = cv2.GaussianBlur(Pv[:, :, c].astype(np.float32), ksize=(0, 0), sigmaX=float(smooth_sigma)).astype(
            np.float64
        )

    def lerp(a0: np.ndarray, a1: np.ndarray, t: float) -> np.ndarray:
        return (1.0 - t) * a0 + t * a1

    out = np.empty((query_xy.shape[0], 2), dtype=np.float64)
    for n in range(query_xy.shape[0]):
        x, y = float(query_xy[n, 0]), float(query_xy[n, 1])
        tx = (x - xmin) / (xmax - xmin)
        ty = (y - ymin) / (ymax - ymin)
        tx = float(np.clip(tx, 0.0, 1.0))
        ty = float(np.clip(ty, 0.0, 1.0))
        fx = tx * (nx - 1)
        fy = ty * (ny - 1)
        i0 = int(np.floor(fx))
        j0 = int(np.floor(fy))
        i1 = min(i0 + 1, nx - 1)
        j1 = min(j0 + 1, ny - 1)
        ax = fx - i0
        ay = fy - j0

        Pu0 = lerp(Pu[j0, i0], Pu[j0, i1], ax)
        Pu1 = lerp(Pu[j1, i0], Pu[j1, i1], ax)
        Pv0 = lerp(Pv[j0, i0], Pv[j0, i1], ax)
        Pv1 = lerp(Pv[j1, i0], Pv[j1, i1], ax)
        au = lerp(Pu0, Pu1, ay)
        av = lerp(Pv0, Pv1, ay)

        qh = np.array([x, y, 1.0], dtype=np.float64)
        out[n, 0] = float(qh @ au)
        out[n, 1] = float(qh @ av)

    return out


def _predict_points_rayfield(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    grid_size: tuple[int, int] = (16, 10),
    smooth_lambda: float = 3.0,
    huber_c: float = 3.0,
    iters: int = 3,
) -> np.ndarray:
    """
    Smooth ray-field (2D warp) on the board plane.

    Fits a regularized grid warp u(x,y), v(x,y) defined at grid nodes in object space.
    This is equivalent to a low-frequency non-parametric camera model restricted to
    the calibration plane.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    import cv2  # type: ignore

    N = int(obj_xy.shape[0])
    if N < 8:
        return _predict_points_mls_homography(obj_xy, img_uv, query_xy)

    nx, ny = (int(grid_size[0]), int(grid_size[1]))
    nx = max(3, nx)
    ny = max(3, ny)
    M = nx * ny

    # Domain covers both constraints and all queries to avoid unstable extrapolation.
    all_xy = np.concatenate([obj_xy, query_xy], axis=0)
    xmin = float(np.min(all_xy[:, 0]))
    xmax = float(np.max(all_xy[:, 0]))
    ymin = float(np.min(all_xy[:, 1]))
    ymax = float(np.max(all_xy[:, 1]))
    if (xmax - xmin) < 1e-9 or (ymax - ymin) < 1e-9:
        return _predict_points_mls_homography(obj_xy, img_uv, query_xy)

    # Pad to avoid boundary artifacts.
    pad_x = 0.05 * (xmax - xmin)
    pad_y = 0.05 * (ymax - ymin)
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    def node_index(ix: int, iy: int) -> int:
        return iy * nx + ix

    def weights_for_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = pts[:, 0]
        y = pts[:, 1]
        tx = (x - xmin) / (xmax - xmin)
        ty = (y - ymin) / (ymax - ymin)
        tx = np.clip(tx, 0.0, 1.0)
        ty = np.clip(ty, 0.0, 1.0)
        fx = tx * (nx - 1)
        fy = ty * (ny - 1)
        i0 = np.floor(fx).astype(np.int32)
        j0 = np.floor(fy).astype(np.int32)
        i1 = np.minimum(i0 + 1, nx - 1)
        j1 = np.minimum(j0 + 1, ny - 1)
        ax = fx - i0
        ay = fy - j0
        w00 = (1.0 - ax) * (1.0 - ay)
        w10 = ax * (1.0 - ay)
        w01 = (1.0 - ax) * ay
        w11 = ax * ay
        idx00 = (j0 * nx + i0).astype(np.int32)
        idx10 = (j0 * nx + i1).astype(np.int32)
        idx01 = (j1 * nx + i0).astype(np.int32)
        idx11 = (j1 * nx + i1).astype(np.int32)
        return idx00, idx10, idx01, idx11, np.stack([w00, w10, w01, w11], axis=1)

    # Smoothness: graph Laplacian on the grid.
    L = np.zeros((M, M), dtype=np.float64)
    for iy in range(ny):
        for ix in range(nx):
            p = node_index(ix, iy)
            neigh = []
            if ix > 0:
                neigh.append(node_index(ix - 1, iy))
            if ix + 1 < nx:
                neigh.append(node_index(ix + 1, iy))
            if iy > 0:
                neigh.append(node_index(ix, iy - 1))
            if iy + 1 < ny:
                neigh.append(node_index(ix, iy + 1))
            deg = len(neigh)
            L[p, p] += deg
            for q in neigh:
                L[p, q] -= 1.0

    # Base mapping: single homography from all correspondences (gives sane extrapolation).
    Hb, _mask = cv2.findHomography(obj_xy, img_uv, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if Hb is None:
        return _predict_points_mls_homography(obj_xy, img_uv, query_xy)

    def proj(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        ph = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
        uvw = (H @ ph.T).T
        return uvw[:, :2] / (uvw[:, 2:3] + 1e-12)

    base_obs = proj(Hb, obj_xy)
    res_obs = img_uv - base_obs

    idx00, idx10, idx01, idx11, ww = weights_for_points(obj_xy)
    nodes_u = np.zeros((M,), dtype=np.float64)
    nodes_v = np.zeros((M,), dtype=np.float64)

    w_data = np.ones((N,), dtype=np.float64)
    huber_c = float(max(0.25, huber_c))

    for _ in range(int(max(1, iters))):
        AtA = np.zeros((M, M), dtype=np.float64)
        Atu = np.zeros((M,), dtype=np.float64)
        Atv = np.zeros((M,), dtype=np.float64)

        for n in range(N):
            wrow = float(w_data[n])
            if wrow <= 0.0:
                continue
            ids = (int(idx00[n]), int(idx10[n]), int(idx01[n]), int(idx11[n]))
            ws = ww[n]
            u_obs = float(res_obs[n, 0])
            v_obs = float(res_obs[n, 1])

            for a in range(4):
                ia = ids[a]
                wa = float(ws[a])
                Atu[ia] += wrow * wa * u_obs
                Atv[ia] += wrow * wa * v_obs
                for b in range(4):
                    ib = ids[b]
                    wb = float(ws[b])
                    AtA[ia, ib] += wrow * wa * wb

        lam = float(smooth_lambda) * (float(N) / float(M))
        AtA = AtA + lam * (L.T @ L) + (0.1 * lam + 1e-6) * np.eye(M, dtype=np.float64)

        nodes_u = np.linalg.solve(AtA, Atu)
        nodes_v = np.linalg.solve(AtA, Atv)

        # Update robust weights.
        du_pred = (
            nodes_u[idx00] * ww[:, 0] + nodes_u[idx10] * ww[:, 1] + nodes_u[idx01] * ww[:, 2] + nodes_u[idx11] * ww[:, 3]
        )
        dv_pred = (
            nodes_v[idx00] * ww[:, 0] + nodes_v[idx10] * ww[:, 1] + nodes_v[idx01] * ww[:, 2] + nodes_v[idx11] * ww[:, 3]
        )
        r = np.sqrt((du_pred - res_obs[:, 0]) ** 2 + (dv_pred - res_obs[:, 1]) ** 2)
        w_data = np.where(r <= huber_c, 1.0, huber_c / (r + 1e-12))

    # Predict queries: base homography + smoothed residual field.
    base_q = proj(Hb, query_xy)
    q00, q10, q01, q11, qw = weights_for_points(query_xy)
    du = nodes_u[q00] * qw[:, 0] + nodes_u[q10] * qw[:, 1] + nodes_u[q01] * qw[:, 2] + nodes_u[q11] * qw[:, 3]
    dv = nodes_v[q00] * qw[:, 0] + nodes_v[q10] * qw[:, 1] + nodes_v[q01] * qw[:, 2] + nodes_v[q11] * qw[:, 3]
    return (base_q + np.stack([du, dv], axis=1)).astype(np.float64)


def _predict_points_rayfield_tps(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    lam: float = 10.0,
) -> np.ndarray:
    """
    Ray-field variant: global homography + TPS-smoothed residuals.

    This keeps the "projective base" (good extrapolation) while using a thin-plate
    spline (with smoothing `lam`) to reconstruct a smooth residual field from
    sparse ArUco samples.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    import cv2  # type: ignore

    N = int(obj_xy.shape[0])
    if N < 8:
        return _predict_points_mls_homography(obj_xy, img_uv, query_xy)

    Hb, _mask = cv2.findHomography(obj_xy, img_uv, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if Hb is None:
        return _predict_points_mls_homography(obj_xy, img_uv, query_xy)

    def proj(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        ph = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
        uvw = (H @ ph.T).T
        return uvw[:, :2] / (uvw[:, 2:3] + 1e-12)

    base_obs = proj(Hb, obj_xy)
    res_obs = img_uv - base_obs

    # TPS reconstructs residual field from sparse samples.
    res_q = _predict_points_tps(obj_xy, res_obs, query_xy, lam=float(lam))
    base_q = proj(Hb, query_xy)
    return (base_q + res_q).astype(np.float64)


def _predict_points_tps_irls(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
) -> np.ndarray:
    """
    Thin-plate spline warp (2D->2D) with robust IRLS weights.

    This implements a weighted TPS smoothing spline where the diagonal regularization
    term becomes `lam * W^{-1}`. With IRLS, weights are updated from the pointwise
    residuals using a Huber rule, down-weighting outliers.

    Notes:
      - For W=I this reduces to the classic `K + lam I` smoothing used by `_predict_points_tps`.
      - This is intended for sparse correspondences with occasional outliers.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    N = int(obj_xy.shape[0])
    if N < 6:
        return _predict_points_mls_affine(obj_xy, img_uv, query_xy)

    m = np.mean(obj_xy, axis=0)
    d = np.sqrt(np.sum((obj_xy - m[None, :]) ** 2, axis=1))
    s = float(np.median(d) + 1e-12)
    X = (obj_xy - m[None, :]) / s
    Q = (query_xy - m[None, :]) / s

    def U(r2: np.ndarray) -> np.ndarray:
        # U(r) = r^2 log(r^2), with U(0)=0.
        r2 = np.asarray(r2, dtype=np.float64)
        out = np.zeros_like(r2)
        mask = r2 > 1e-18
        out[mask] = r2[mask] * np.log(r2[mask])
        return out

    dx = X[:, 0:1] - X[:, 0:1].T
    dy = X[:, 1:2] - X[:, 1:2].T
    K = U(dx * dx + dy * dy)
    P = np.concatenate([np.ones((N, 1), dtype=np.float64), X], axis=1)  # (N,3)

    w_data = np.ones((N,), dtype=np.float64)
    huber_c = float(max(0.25, huber_c))
    lam = float(max(0.0, lam))

    coeff = None
    eps = 1e-12
    for _ in range(int(max(1, iters))):
        # Weighted smoothing spline: (K + lam * W^{-1}) w + P a = y;  P^T w = 0.
        D = lam / (w_data + eps)
        A = np.zeros((N + 3, N + 3), dtype=np.float64)
        A[:N, :N] = K + np.diag(D)
        A[:N, N:] = P
        A[N:, :N] = P.T

        Y = np.zeros((N + 3, 2), dtype=np.float64)
        Y[:N, :] = img_uv

        try:
            coeff = np.linalg.solve(A, Y)  # (N+3,2): [W; a0,a1,a2]
        except np.linalg.LinAlgError:
            return _predict_points_mls_affine(obj_xy, img_uv, query_xy)

        Wc = coeff[:N, :]
        ac = coeff[N:, :]  # (3,2)

        # Robust weights from residuals on the constraints.
        pred_i = K @ Wc + P @ ac
        r = np.sqrt(np.sum((pred_i - img_uv) ** 2, axis=1))
        w_data = np.where(r <= huber_c, 1.0, huber_c / (r + eps))

    if coeff is None:
        return _predict_points_mls_affine(obj_xy, img_uv, query_xy)

    Wc = coeff[:N, :]
    ac = coeff[N:, :]
    dxq = Q[:, 0:1] - X[:, 0:1].T  # (M,N)
    dyq = Q[:, 1:2] - X[:, 1:2].T
    Kq = U(dxq * dxq + dyq * dyq)  # (M,N)
    Pq = np.concatenate([np.ones((Q.shape[0], 1), dtype=np.float64), Q], axis=1)  # (M,3)
    return Kq @ Wc + Pq @ ac


def _predict_points_rayfield_tps_robust(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
) -> np.ndarray:
    """
    Ray-field variant: global homography + robust TPS-smoothed residuals (IRLS).
    """
    from stereocomplex.core.rayfield2d import predict_points_rayfield_tps_robust  # noqa: PLC0415

    return predict_points_rayfield_tps_robust(
        obj_xy,
        img_uv,
        query_xy,
        lam=float(lam),
        huber_c=float(huber_c),
        iters=int(iters),
    )


def _predict_points_mls_homography(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    k: int = 60,
    sigma_obj: float | None = None,
) -> np.ndarray:
    """
    Moving least squares (projective) mapping from board coords -> image coords.

    Fits a local homography per query point using weighted DLT on the k strongest
    correspondences in object space.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    N = int(obj_xy.shape[0])
    k = int(max(8, min(k, N)))
    if sigma_obj is None:
        span = float(np.median(np.sqrt(np.sum((obj_xy - np.median(obj_xy, axis=0)) ** 2, axis=1))) + 1e-9)
        sigma_obj = max(10.0, 0.25 * span)
    sigma2 = float(sigma_obj) ** 2

    def _normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.mean(pts, axis=0)
        d = np.sqrt(np.sum((pts - m[None, :]) ** 2, axis=1))
        s = float(np.sqrt(2.0) / (np.mean(d) + 1e-12))
        T = np.array([[s, 0.0, -s * m[0]], [0.0, s, -s * m[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
        ph = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
        pn = (T @ ph.T).T[:, :2]
        return pn, T

    # Global fallback homography (helps for queries outside convex hull).
    H_global = None
    if N >= 4:
        A = []
        for (x, y), (u, v) in zip(obj_xy.tolist(), img_uv.tolist(), strict=True):
            A.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
            A.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
        A = np.asarray(A, dtype=np.float64)
        _u, _s, vt = np.linalg.svd(A, full_matrices=False)
        H_global = vt[-1].reshape(3, 3)

    out = np.empty((query_xy.shape[0], 2), dtype=np.float64)
    for i in range(query_xy.shape[0]):
        q = query_xy[i]
        d2 = np.sum((obj_xy - q[None, :]) ** 2, axis=1)
        w = np.exp(-0.5 * d2 / sigma2)
        if N > k:
            idx = np.argpartition(w, -k)[-k:]
        else:
            idx = np.arange(N)
        ww = w[idx]
        if float(np.max(ww)) <= 1e-12:
            if H_global is None:
                out[i] = np.array([np.nan, np.nan], dtype=np.float64)
            else:
                uvw = H_global @ np.array([q[0], q[1], 1.0], dtype=np.float64)
                out[i] = uvw[:2] / (uvw[2] + 1e-12)
            continue

        X = obj_xy[idx]
        U = img_uv[idx]
        Xn, T1 = _normalize(X)
        Un, T2 = _normalize(U)

        A = np.zeros((2 * idx.shape[0], 9), dtype=np.float64)
        for j, ((x, y), (u, v)) in enumerate(zip(Xn.tolist(), Un.tolist(), strict=True)):
            A[2 * j + 0] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
            A[2 * j + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]
        sw = np.sqrt(np.repeat(ww, 2))
        A *= sw[:, None]

        _u, _s, vt = np.linalg.svd(A, full_matrices=False)
        Hn = vt[-1].reshape(3, 3)
        H = np.linalg.inv(T2) @ Hn @ T1
        uvw = H @ np.array([q[0], q[1], 1.0], dtype=np.float64)
        out[i, 0] = float(uvw[0] / (uvw[2] + 1e-12))
        out[i, 1] = float(uvw[1] / (uvw[2] + 1e-12))

    return out


def _predict_points_piecewise_affine(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
) -> np.ndarray:
    """
    Piecewise-affine mapping using Delaunay triangulation in object space.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    import cv2  # type: ignore

    xmin = float(np.min(obj_xy[:, 0]))
    ymin = float(np.min(obj_xy[:, 1]))
    xmax = float(np.max(obj_xy[:, 0]))
    ymax = float(np.max(obj_xy[:, 1]))
    pad = 1.0
    rect = (int(np.floor(xmin - pad)), int(np.floor(ymin - pad)), int(np.ceil((xmax - xmin) + 2 * pad)), int(np.ceil((ymax - ymin) + 2 * pad)))
    subdiv = cv2.Subdiv2D(rect)
    for p in obj_xy.tolist():
        subdiv.insert((float(p[0]), float(p[1])))
    tris = subdiv.getTriangleList()
    tris = np.asarray(tris, dtype=np.float64).reshape(-1, 6)

    def _closest_index(pt: np.ndarray) -> int | None:
        d2 = np.sum((obj_xy - pt[None, :]) ** 2, axis=1)
        j = int(np.argmin(d2))
        if float(d2[j]) > 1e-8:
            return None
        return j

    tri_idx: list[tuple[int, int, int]] = []
    for t in tris:
        p1 = np.array([t[0], t[1]], dtype=np.float64)
        p2 = np.array([t[2], t[3]], dtype=np.float64)
        p3 = np.array([t[4], t[5]], dtype=np.float64)
        i1 = _closest_index(p1)
        i2 = _closest_index(p2)
        i3 = _closest_index(p3)
        if i1 is None or i2 is None or i3 is None:
            continue
        if len({i1, i2, i3}) < 3:
            continue
        tri_idx.append((i1, i2, i3))
    if not tri_idx:
        return _predict_points_mls_affine(obj_xy, img_uv, query_xy)

    # Precompute triangles in both spaces.
    obj_tris = []
    img_tris = []
    for a, b, c in tri_idx:
        obj_tris.append(obj_xy[[a, b, c], :])
        img_tris.append(img_uv[[a, b, c], :])

    out = np.empty((query_xy.shape[0], 2), dtype=np.float64)
    eps = -1e-8
    for i, q in enumerate(query_xy):
        found = False
        for P, Q in zip(obj_tris, img_tris, strict=True):
            p0, p1, p2 = P
            v0 = p1 - p0
            v1 = p2 - p0
            v2 = q - p0
            den = float(v0[0] * v1[1] - v0[1] * v1[0])
            if abs(den) < 1e-12:
                continue
            a = float((v2[0] * v1[1] - v2[1] * v1[0]) / den)
            b = float((v0[0] * v2[1] - v0[1] * v2[0]) / den)
            c = 1.0 - a - b
            if a >= eps and b >= eps and c >= eps:
                out[i] = a * Q[1] + b * Q[2] + c * Q[0]
                found = True
                break
        if not found:
            out[i] = _predict_points_mls_affine(obj_xy, img_uv, q.reshape(1, 2))[0]
    return out


def _predict_points_tps(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    lam: float = 1e-1,
) -> np.ndarray:
    """
    Thin-plate spline warp (2D->2D) from board coords -> image coords.

    Uses only 2D correspondences (e.g., ArUco marker corners), so it can model
    non-pinhole / non-Brown camera mappings as a smooth deformation.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    N = int(obj_xy.shape[0])
    if N < 6:
        # Not enough constraints; fall back to affine MLS.
        return _predict_points_mls_affine(obj_xy, img_uv, query_xy)

    m = np.mean(obj_xy, axis=0)
    d = np.sqrt(np.sum((obj_xy - m[None, :]) ** 2, axis=1))
    s = float(np.median(d) + 1e-12)
    X = (obj_xy - m[None, :]) / s
    Q = (query_xy - m[None, :]) / s

    def U(r2: np.ndarray) -> np.ndarray:
        # U(r) = r^2 log(r^2), with U(0)=0.
        r2 = np.asarray(r2, dtype=np.float64)
        out = np.zeros_like(r2)
        mask = r2 > 1e-18
        out[mask] = r2[mask] * np.log(r2[mask])
        return out

    dx = X[:, 0:1] - X[:, 0:1].T
    dy = X[:, 1:2] - X[:, 1:2].T
    K = U(dx * dx + dy * dy)
    if lam > 0:
        K = K + float(lam) * np.eye(N, dtype=np.float64)

    P = np.concatenate([np.ones((N, 1), dtype=np.float64), X], axis=1)  # (N,3)
    A = np.zeros((N + 3, N + 3), dtype=np.float64)
    A[:N, :N] = K
    A[:N, N:] = P
    A[N:, :N] = P.T

    Y = np.zeros((N + 3, 2), dtype=np.float64)
    Y[:N, :] = img_uv

    try:
        coeff = np.linalg.solve(A, Y)  # (N+3,2): [W; a0,a1,a2]
    except np.linalg.LinAlgError:
        return _predict_points_mls_affine(obj_xy, img_uv, query_xy)

    W = coeff[:N, :]
    a = coeff[N:, :]  # (3,2)

    # Compute warp for queries.
    dxq = Q[:, 0:1] - X[:, 0:1].T  # (M,N)
    dyq = Q[:, 1:2] - X[:, 1:2].T
    Kq = U(dxq * dxq + dyq * dyq)  # (M,N)
    Pq = np.concatenate([np.ones((Q.shape[0], 1), dtype=np.float64), Q], axis=1)  # (M,3)
    return Kq @ W + Pq @ a


def _bilinear(img: np.ndarray, x: float, y: float) -> float:
    H, W = img.shape[:2]
    if x < 0.0 or y < 0.0 or x > (W - 1) or y > (H - 1):
        return 0.0
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)
    wx = x - x0
    wy = y - y0
    Ia = float(img[y0, x0])
    Ib = float(img[y0, x1])
    Ic = float(img[y1, x0])
    Id = float(img[y1, x1])
    return (1.0 - wx) * (1.0 - wy) * Ia + wx * (1.0 - wy) * Ib + (1.0 - wx) * wy * Ic + wx * wy * Id
