from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stereocomplex.eval.detectors.charuco import detect_image_features
from stereocomplex.eval.predictors.dispatch import predict_charuco_points
from stereocomplex.eval.refiners.dispatch import refine_detected_points

LOGGER = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class _SceneEvalContext:
    frames: list[dict[str, object]]
    frame_id: np.ndarray
    gt_by_frame: dict[int, dict[str, dict[int, np.ndarray]]]
    cv2: object
    aruco: object
    dictionary: object
    charuco_board: object
    detector_params: object
    aruco_detector: object
    charuco_detector: object
    K_left: np.ndarray
    d_left: np.ndarray
    K_right: np.ndarray
    d_right: np.ndarray


def eval_charuco_detection(
    dataset_root: Path,
    write_json: bool = True,
    method: str = "charuco",
    refine: str = "none",
    tensor_sigma: float = 1.5,
    search_radius: int = 3,
) -> None:
    """Evaluate ChArUco corner detection accuracy across all scenes in a dataset.

    Parameters
    ----------
    dataset_root : Path
        Root directory containing ``train/``, ``val/``, ``test/`` subdirectories.
    write_json : bool
        If True, write the aggregate report to ``charuco_detection_report.json``.
    method : str
        Corner identification method (see ``collect_charuco_scene_errors``).
    refine : str
        Subpixel refinement strategy.
    tensor_sigma : float
        Gaussian sigma for tensor-based refinement.
    search_radius : int
        Search window half-width for refinement.
    """
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
            LOGGER.info("%s/%s: %s", split, scene_dir.name, json.dumps(scene_stats, sort_keys=True))

    if write_json:
        out = dataset_root / "charuco_detection_report.json"
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        LOGGER.info("Wrote %s", out)


def eval_charuco_scene(
    scene_dir: Path,
    method: str = "charuco",
    refine: str = "none",
    tensor_sigma: float = 1.5,
    search_radius: int = 3,
) -> dict[str, object]:
    """Evaluate ChArUco corner detection accuracy for a single scene.

    Parameters
    ----------
    scene_dir : Path
        Scene directory containing ``meta.json``, ``frames.jsonl``,
        ``gt_charuco_corners.npz``, and ``left/`` / ``right/`` image subdirs.
    method : str
        Corner identification method.
    refine : str
        Subpixel refinement strategy.
    tensor_sigma : float
        Gaussian sigma for tensor-based refinement.
    search_radius : int
        Search window half-width for refinement.

    Returns
    -------
    dict
        Per-side error statistics (n_frames, n_matched, rms_px, p95_px, etc.).
    """
    context = _load_scene_eval_context(scene_dir)

    per_side_errors: dict[str, list[float]] = {"left": [], "right": []}
    per_side_dx: dict[str, list[float]] = {"left": [], "right": []}
    per_side_dy: dict[str, list[float]] = {"left": [], "right": []}
    per_side_n_detected: dict[str, int] = {"left": 0, "right": 0}
    per_side_n_matched: dict[str, int] = {"left": 0, "right": 0}

    n_frames = 0
    n_frames_with_any_match = 0

    for fr in context.frames:
        n_frames += 1
        fid = int(fr["frame_id"])
        gt_frame = context.gt_by_frame.get(fid)
        if gt_frame is None:
            continue

        (eL, dxL, dyL, n_det_L, n_match_L), (eR, dxR, dyR, n_det_R, n_match_R) = _eval_image_pair(
            scene_dir,
            context,
            fr,
            gt_frame,
            method=method,
            refine=refine,
            tensor_sigma=tensor_sigma,
            search_radius=search_radius,
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
        "n_gt_rows": int(context.frame_id.shape[0]),
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
    context = _load_scene_eval_context(scene_dir)

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

    for fr in context.frames:
        fid = int(fr["frame_id"])
        gt_frame = context.gt_by_frame.get(fid)
        if gt_frame is None:
            continue

        (eL, dxL, dyL, n_det_L, n_match_L), (eR, dxR, dyR, n_det_R, n_match_R) = _eval_image_pair(
            scene_dir,
            context,
            fr,
            gt_frame,
            method=method,
            refine=refine,
            tensor_sigma=tensor_sigma,
            search_radius=search_radius,
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
        "left": {
            "n_detected": n_det_left,
            "n_matched": n_match_left,
            "errors": errors_left,
            "dx": dx_left,
            "dy": dy_left,
        },
        "right": {
            "n_detected": n_det_right,
            "n_matched": n_match_right,
            "errors": errors_right,
            "dx": dx_right,
            "dy": dy_right,
        },
    }


def _load_scene_eval_context(scene_dir: Path) -> _SceneEvalContext:
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

    return _SceneEvalContext(
        frames=frames,
        frame_id=frame_id,
        gt_by_frame=gt_by_frame,
        cv2=cv2,
        aruco=aruco,
        dictionary=dictionary,
        charuco_board=charuco_board,
        detector_params=detector_params,
        aruco_detector=aruco_detector,
        charuco_detector=charuco_detector,
        K_left=K_left,
        d_left=d_left,
        K_right=K_right,
        d_right=d_right,
    )


def _eval_image_pair(
    scene_dir: Path,
    context: _SceneEvalContext,
    frame: dict[str, object],
    gt_frame: dict[str, dict[int, np.ndarray]],
    *,
    method: str,
    refine: str,
    tensor_sigma: float,
    search_radius: int,
):
    left_path = scene_dir / "left" / str(frame["left"])
    right_path = scene_dir / "right" / str(frame["right"])

    left = _eval_one_image(
        context.cv2,
        context.aruco,
        context.dictionary,
        context.charuco_board,
        context.detector_params,
        context.aruco_detector,
        context.charuco_detector,
        left_path,
        gt_frame,
        side="left",
        method=method,
        refine=refine,
        tensor_sigma=tensor_sigma,
        search_radius=search_radius,
        camera_matrix=context.K_left,
        dist_coeffs=context.d_left,
    )
    right = _eval_one_image(
        context.cv2,
        context.aruco,
        context.dictionary,
        context.charuco_board,
        context.detector_params,
        context.aruco_detector,
        context.charuco_detector,
        right_path,
        gt_frame,
        side="right",
        method=method,
        refine=refine,
        tensor_sigma=tensor_sigma,
        search_radius=search_radius,
        camera_matrix=context.K_right,
        dist_coeffs=context.d_right,
    )
    return left, right


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
    for fid, cid, left_uv, right_uv in zip(frame_id.tolist(), corner_id.tolist(), uvL, uvR, strict=True):
        f = out.setdefault(int(fid), {"left": {}, "right": {}})
        f["left"][int(cid)] = left_uv
        f["right"][int(cid)] = right_uv
    return out


def _make_charuco_detector(board_meta: dict):
    try:
        import cv2  # type: ignore
        from cv2 import aruco  # type: ignore
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

    features, n_detected_on_failure = detect_image_features(
        aruco,
        dictionary,
        charuco_board,
        detector_params,
        aruco_detector,
        charuco_detector,
        img,
        method,
    )
    if features is None:
        return [], [], [], n_detected_on_failure, 0

    predicted = predict_charuco_points(
        cv2,
        charuco_board,
        features,
        method,
        camera_matrix,
        dist_coeffs,
    )
    if predicted is None:
        return [], [], [], 0, 0
    charuco_ids, charuco_xy = predicted

    charuco_xy = refine_detected_points(
        refine,
        cv2,
        img,
        charuco_xy,
        search_radius=search_radius,
        tensor_sigma=tensor_sigma,
    )

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
