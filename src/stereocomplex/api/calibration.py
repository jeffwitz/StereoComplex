from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from stereocomplex.api.corner_refinement import CharucoDetections, RefineMethod, refine_charuco_corners
from stereocomplex.api.model_io import save_stereo_central_rayfield
from stereocomplex.api.stereo_reconstruction import StereoCentralRayFieldModel
from stereocomplex.core.image_io import load_gray_u8
from stereocomplex.ray3d.central_ba import FrameObservations
from stereocomplex.ray3d.central_stereo_ba import StereoFrameObservations, fit_central_stereo_rayfield_ba


@dataclass(frozen=True)
class CharucoBoardSpec:
    squares_x: int
    squares_y: int
    square_size_mm: float
    marker_size_mm: float
    aruco_dictionary: str = "DICT_4X4_1000"
    adaptive_thresh_win_size_max: int | None = None
    corner_refinement_win_size: int = 5
    corner_refinement_max_iterations: int = 50
    corner_refinement_min_accuracy: float = 1e-3
    check_markers: bool | None = None
    min_markers: int | None = None
    try_refine_markers: bool | None = None
    legacy_pattern: bool = False  # for pre-4.x OpenCV ChArUco patterns

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CharucoBoardSpec":
        return cls(
            squares_x=int(payload["squares_x"]),
            squares_y=int(payload["squares_y"]),
            square_size_mm=float(payload["square_size_mm"]),
            marker_size_mm=float(payload["marker_size_mm"]),
            aruco_dictionary=str(payload.get("aruco_dictionary", "DICT_4X4_1000")),
            adaptive_thresh_win_size_max=(
                int(payload["adaptive_thresh_win_size_max"])
                if payload.get("adaptive_thresh_win_size_max") is not None
                else None
            ),
            corner_refinement_win_size=int(payload.get("corner_refinement_win_size", 5)),
            corner_refinement_max_iterations=int(payload.get("corner_refinement_max_iterations", 50)),
            corner_refinement_min_accuracy=float(payload.get("corner_refinement_min_accuracy", 1e-3)),
            check_markers=bool(payload["check_markers"]) if payload.get("check_markers") is not None else None,
            min_markers=int(payload["min_markers"]) if payload.get("min_markers") is not None else None,
            try_refine_markers=(
                bool(payload["try_refine_markers"]) if payload.get("try_refine_markers") is not None else None
            ),
            legacy_pattern=bool(payload.get("legacy_pattern", False)),
        )

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "CharucoBoardSpec":
        board_meta = meta["board"]
        opencv_meta = meta.get("opencv", {})
        opencv_aruco = opencv_meta.get("aruco_detector", {}) if isinstance(opencv_meta, dict) else {}
        opencv_charuco = opencv_meta.get("charuco_detector", {}) if isinstance(opencv_meta, dict) else {}
        return cls(
            squares_x=int(board_meta["squares_x"]),
            squares_y=int(board_meta["squares_y"]),
            square_size_mm=float(board_meta["square_size_mm"]),
            marker_size_mm=float(board_meta["marker_size_mm"]),
            aruco_dictionary=str(board_meta.get("aruco_dictionary", "DICT_4X4_1000")),
            adaptive_thresh_win_size_max=(
                int(opencv_aruco["adaptiveThreshWinSizeMax"])
                if isinstance(opencv_aruco, dict) and opencv_aruco.get("adaptiveThreshWinSizeMax") is not None
                else None
            ),
            corner_refinement_win_size=int(opencv_aruco.get("cornerRefinementWinSize", 5))
            if isinstance(opencv_aruco, dict)
            else 5,
            corner_refinement_max_iterations=int(opencv_aruco.get("cornerRefinementMaxIterations", 50))
            if isinstance(opencv_aruco, dict)
            else 50,
            corner_refinement_min_accuracy=float(opencv_aruco.get("cornerRefinementMinAccuracy", 1e-3))
            if isinstance(opencv_aruco, dict)
            else 1e-3,
            check_markers=(
                bool(opencv_charuco["checkMarkers"])
                if isinstance(opencv_charuco, dict) and opencv_charuco.get("checkMarkers") is not None
                else None
            ),
            min_markers=(
                int(opencv_charuco["minMarkers"])
                if isinstance(opencv_charuco, dict) and opencv_charuco.get("minMarkers") is not None
                else None
            ),
            try_refine_markers=(
                bool(opencv_charuco["tryRefineMarkers"])
                if isinstance(opencv_charuco, dict) and opencv_charuco.get("tryRefineMarkers") is not None
                else None
            ),
            legacy_pattern=bool(opencv_charuco.get("legacyPattern", False)),
        )


@dataclass(frozen=True)
class StereoImagePair:
    left_path: Path
    right_path: Path
    frame_id: int | None = None


@dataclass(frozen=True)
class StereoCentralRayFieldFitReport:
    image_width_px: int
    image_height_px: int
    n_input_pairs: int
    n_detected_pairs: int
    n_observation_frames: int
    n_initialized_frames: int
    used_frame_ids: tuple[int, ...]
    method2d: RefineMethod
    min_common_corners: int
    nmax: int
    train_skew_rms_mm: float
    train_skew_p95_mm: float
    train_point_to_ray_rms_mm: float
    train_point_to_ray_p95_mm: float
    n_points_total: int
    mean_common_corners_per_frame: float
    diagnostics: dict[str, float]
    exported_model_json: str | None = None


@dataclass(frozen=True)
class StereoCentralRayFieldFitResult:
    model: StereoCentralRayFieldModel
    report: StereoCentralRayFieldFitReport


@dataclass(frozen=True)
class StereoOpenCVCalibrationReport:
    image_width_px: int
    image_height_px: int
    n_input_pairs: int
    n_detected_pairs: int
    n_mono_frames_left: int
    n_mono_frames_right: int
    n_stereo_frames: int
    used_frame_ids: tuple[int, ...]
    method2d: RefineMethod
    min_common_corners: int
    mono_left_rms_px: float
    mono_right_rms_px: float
    stereo_rms_px: float
    baseline_mm: float


@dataclass(frozen=True)
class StereoOpenCVCalibrationResult:
    K_left: np.ndarray
    dist_left: np.ndarray
    K_right: np.ndarray
    dist_right: np.ndarray
    R_right_from_left: np.ndarray
    t_right_from_left_mm: np.ndarray
    essential_matrix: np.ndarray
    fundamental_matrix: np.ndarray
    report: StereoOpenCVCalibrationReport

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary."""
        return {
            "K_left": self.K_left.tolist(),
            "dist_left": self.dist_left.tolist(),
            "K_right": self.K_right.tolist(),
            "dist_right": self.dist_right.tolist(),
            "R": self.R_right_from_left.tolist(),
            "T_mm": self.t_right_from_left_mm.tolist(),
            "stereo_rms_px": float(self.report.stereo_rms_px),
            "mono_rms_left_px": float(self.report.mono_rms_left_px),
            "mono_rms_right_px": float(self.report.mono_rms_right_px),
            "n_stereo_frames": int(self.report.n_stereo_frames),
        }

    def to_opencv(self) -> tuple:
        """Return ``(K1, d1, K2, d2, R, T)`` as used by OpenCV."""
        return (
            self.K_left, self.dist_left,
            self.K_right, self.dist_right,
            self.R_right_from_left, self.t_right_from_left_mm,
        )


@dataclass(frozen=True)
class _CharucoRuntime:
    cv2: Any
    aruco: Any
    board: Any
    dictionary: Any
    detector_params: Any
    aruco_detector: Any
    charuco_detector: Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_frames(scene_dir: Path) -> list[dict[str, Any]]:
    frames_path = scene_dir / "frames.jsonl"
    frames: list[dict[str, Any]] = []
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


def _build_charuco_runtime(board: CharucoBoardSpec) -> _CharucoRuntime:
    import cv2  # type: ignore
    import cv2.aruco as aruco  # type: ignore

    dict_id = getattr(aruco, str(board.aruco_dictionary), None)
    if dict_id is None:
        raise ValueError(f"Unknown aruco_dictionary: {board.aruco_dictionary}")
    dictionary = aruco.getPredefinedDictionary(dict_id)

    if hasattr(aruco, "CharucoBoard"):
        charuco_board = aruco.CharucoBoard(
            (int(board.squares_x), int(board.squares_y)),
            float(board.square_size_mm),
            float(board.marker_size_mm),
            dictionary,
        )
        if board.legacy_pattern and hasattr(charuco_board, "setLegacyPattern"):
            charuco_board.setLegacyPattern(True)
    elif hasattr(aruco, "CharucoBoard_create"):  # pragma: no cover
        charuco_board = aruco.CharucoBoard_create(
            int(board.squares_x),
            int(board.squares_y),
            float(board.square_size_mm),
            float(board.marker_size_mm),
            dictionary,
        )
    else:  # pragma: no cover
        raise RuntimeError("cv2.aruco does not expose CharucoBoard APIs in this build.")

    detector_params = aruco.DetectorParameters()
    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = int(board.corner_refinement_win_size)
    detector_params.cornerRefinementMaxIterations = int(board.corner_refinement_max_iterations)
    detector_params.cornerRefinementMinAccuracy = float(board.corner_refinement_min_accuracy)
    if board.adaptive_thresh_win_size_max is not None:
        detector_params.adaptiveThreshWinSizeMax = int(board.adaptive_thresh_win_size_max)
    # Wider detection range for small markers (e.g. Pycaso 0.15 mm markers)
    if hasattr(detector_params, "minMarkerPerimeterRate"):
        detector_params.minMarkerPerimeterRate = 0.005
    if hasattr(detector_params, "maxMarkerPerimeterRate"):
        detector_params.maxMarkerPerimeterRate = 0.20
    if hasattr(detector_params, "polygonalApproxAccuracyRate"):
        detector_params.polygonalApproxAccuracyRate = 0.03

    charuco_detector = None
    if hasattr(aruco, "CharucoDetector"):
        charuco_detector = aruco.CharucoDetector(charuco_board)
        if hasattr(charuco_detector, "setDetectorParameters"):
            charuco_detector.setDetectorParameters(detector_params)
        if hasattr(charuco_detector, "getCharucoParameters") and hasattr(charuco_detector, "setCharucoParameters"):
            cp = charuco_detector.getCharucoParameters()
            if board.check_markers is not None:
                cp.checkMarkers = bool(board.check_markers)
            if board.min_markers is not None:
                cp.minMarkers = int(board.min_markers)
            if board.try_refine_markers is not None:
                cp.tryRefineMarkers = bool(board.try_refine_markers)
            charuco_detector.setCharucoParameters(cp)

    aruco_detector = None
    if charuco_detector is None and hasattr(aruco, "ArucoDetector"):
        aruco_detector = aruco.ArucoDetector(dictionary, detector_params)

    return _CharucoRuntime(
        cv2=cv2,
        aruco=aruco,
        board=charuco_board,
        dictionary=dictionary,
        detector_params=detector_params,
        aruco_detector=aruco_detector,
        charuco_detector=charuco_detector,
    )


def build_charuco_board(board: CharucoBoardSpec) -> Any:
    """
    Build an OpenCV `aruco.CharucoBoard` from a stable Python dataclass.
    """
    return _build_charuco_runtime(board).board


def detect_charuco_corners(*, image: str | Path | np.ndarray, board: CharucoBoardSpec) -> CharucoDetections | None:
    """
    Detect ArUco markers and ChArUco corners in one image.

    This is a small convenience wrapper around OpenCV detection so downstream code
    can stay inside the public StereoComplex API.
    """
    runtime = _build_charuco_runtime(board)
    img_gray = _ensure_gray_u8(image)
    return _detect_view(runtime, img_gray)


def _detect_view(runtime: _CharucoRuntime, img_gray: np.ndarray) -> CharucoDetections | None:
    aruco = runtime.aruco

    if runtime.charuco_detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = runtime.charuco_detector.detectBoard(img_gray)
    else:
        if runtime.aruco_detector is not None:
            marker_corners, marker_ids, _rej = runtime.aruco_detector.detectMarkers(img_gray)
        else:  # pragma: no cover
            marker_corners, marker_ids, _rej = aruco.detectMarkers(
                img_gray, runtime.dictionary, parameters=runtime.detector_params
            )

        charuco_corners, charuco_ids = None, None
        if hasattr(aruco, "interpolateCornersCharuco") and marker_ids is not None and len(marker_ids) > 0:
            ret = aruco.interpolateCornersCharuco(marker_corners, marker_ids, img_gray, runtime.board)
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
        charuco_xy = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2) - 0.5

    return CharucoDetections(
        marker_ids=marker_ids_arr,
        marker_corners=marker_corners_arr,
        charuco_ids=charuco_ids_arr,
        charuco_xy=charuco_xy,
    )


def _dict_from_ids_xy(ids: np.ndarray, xy: np.ndarray) -> dict[int, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return {int(i): xy[k].astype(np.float64) for k, i in enumerate(ids.tolist())}


def _refine_detection_points(
    *,
    runtime: _CharucoRuntime,
    detection: CharucoDetections,
    method2d: RefineMethod,
    tps_lam: float,
    huber_c: float,
    iters: int,
) -> np.ndarray:
    if method2d == "raw":
        return np.asarray(detection.charuco_xy, dtype=np.float64).reshape(-1, 2)
    return np.asarray(
        refine_charuco_corners(
            method=str(method2d),
            board=runtime.board,
            marker_ids=detection.marker_ids,
            marker_corners=detection.marker_corners,
            charuco_ids=detection.charuco_ids,
            charuco_xy=detection.charuco_xy,
            tps_lam=float(tps_lam),
            huber_c=float(huber_c),
            iters=int(iters),
        ),
        dtype=np.float64,
    ).reshape(-1, 2)


def _estimate_K0_from_homographies(*, homographies: list[np.ndarray], image_size: tuple[int, int]) -> np.ndarray:
    w, h = int(image_size[0]), int(image_size[1])
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    K_fallback = np.array([[1.5 * float(max(w, h)), 0.0, cx], [0.0, 1.5 * float(max(w, h)), cy], [0.0, 0.0, 1.0]])

    Hs: list[np.ndarray] = []
    for H in homographies:
        H = np.asarray(H, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(H)):
            continue
        if abs(float(H[2, 2])) < 1e-12:
            continue
        Hs.append(H / float(H[2, 2]))
    if len(Hs) < 3:
        return K_fallback.astype(np.float64)

    f_min = 0.5 * float(max(w, h))
    f_max = 3.0 * float(max(w, h))
    fs = np.logspace(np.log10(f_min), np.log10(f_max), num=80, dtype=np.float64)

    def cost_for_f(f: float) -> float:
        Kinv = np.array([[1.0 / f, 0.0, -cx / f], [0.0, 1.0 / f, -cy / f], [0.0, 0.0, 1.0]], dtype=np.float64)
        c = 0.0
        for H in Hs:
            Hn = Kinv @ H
            h1 = Hn[:, 0]
            h2 = Hn[:, 1]
            n1 = float(np.linalg.norm(h1))
            n2 = float(np.linalg.norm(h2))
            if not np.isfinite(n1) or not np.isfinite(n2) or n1 < 1e-12 or n2 < 1e-12:
                continue
            dot = float(np.dot(h1, h2)) / (n1 * n2)
            ratio = (n1 / n2) - 1.0
            c += dot * dot + ratio * ratio
        return c

    costs = np.array([cost_for_f(float(f)) for f in fs], dtype=np.float64)
    if np.all(np.isfinite(costs)):
        f0 = float(fs[int(np.argmin(costs))])
        if np.isfinite(f0) and f0 > 1.0:
            return np.array([[f0, 0.0, cx], [0.0, f0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def v_ij(H: np.ndarray, i: int, j: int) -> np.ndarray:
        h_i = H[:, i].reshape(3)
        h_j = H[:, j].reshape(3)
        return np.array(
            [
                h_i[0] * h_j[0],
                h_i[0] * h_j[1] + h_i[1] * h_j[0],
                h_i[1] * h_j[1],
                h_i[2] * h_j[0] + h_i[0] * h_j[2],
                h_i[2] * h_j[1] + h_i[1] * h_j[2],
                h_i[2] * h_j[2],
            ],
            dtype=np.float64,
        )

    V_rows: list[np.ndarray] = []
    for H in Hs:
        V_rows.append(v_ij(H, 0, 1))
        V_rows.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))
    V = np.stack(V_rows, axis=0)
    _U, _S, Vt = np.linalg.svd(V)
    b = Vt[-1, :].reshape(6)
    if not np.all(np.isfinite(b)):
        return K_fallback.astype(np.float64)

    b11, b12, b22, b13, b23, b33 = (float(x) for x in b.tolist())
    denom = b11 * b22 - b12 * b12
    if not np.isfinite(denom) or abs(denom) < 1e-18:
        return K_fallback.astype(np.float64)

    v0 = (b12 * b13 - b11 * b23) / denom
    lam = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11
    if not np.isfinite(lam) or lam <= 0:
        return K_fallback.astype(np.float64)

    alpha = np.sqrt(lam / b11)
    beta = np.sqrt(lam * b11 / denom)
    gamma = -b12 * alpha * alpha * beta / lam
    u0 = gamma * v0 / beta - b13 * alpha * alpha / lam
    if not all(np.isfinite(x) for x in [alpha, beta, gamma, u0, v0]):
        return K_fallback.astype(np.float64)
    if alpha <= 1e-6 or beta <= 1e-6:
        return K_fallback.astype(np.float64)

    K0 = np.array([[alpha, gamma, u0], [0.0, beta, v0], [0.0, 0.0, 1.0]], dtype=np.float64)
    K0[0, 2] = float(np.clip(K0[0, 2], cx - 0.5 * w, cx + 0.5 * w))
    K0[1, 2] = float(np.clip(K0[1, 2], cy - 0.5 * h, cy + 0.5 * h))
    return K0


def _init_pose_from_homography(
    cv2: Any,
    *,
    obj_xy_mm: np.ndarray,
    uv_px: np.ndarray,
    K0: np.ndarray,
    ransac_thresh_px: float = 3.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    obj_xy_mm = np.asarray(obj_xy_mm, dtype=np.float64).reshape(-1, 2)
    uv_px = np.asarray(uv_px, dtype=np.float64).reshape(-1, 2)
    if obj_xy_mm.shape[0] < 6:
        return None

    H, _mask = cv2.findHomography(
        obj_xy_mm,
        uv_px,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_thresh_px),
    )
    if H is None:
        return None

    K0 = np.asarray(K0, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(K0)):
        return None
    if abs(float(K0[2, 2]) - 1.0) > 1e-6:
        K0 = K0 / float(K0[2, 2])
    Hn = np.linalg.inv(K0) @ H
    h1 = Hn[:, 0]
    h2 = Hn[:, 1]
    h3 = Hn[:, 2]
    s1 = np.linalg.norm(h1)
    s2 = np.linalg.norm(h2)
    if not np.isfinite(s1) or not np.isfinite(s2) or s1 < 1e-12 or s2 < 1e-12:
        return None
    s = 1.0 / (0.5 * (s1 + s2))
    r1 = s * h1
    r2 = s * h2
    r3 = np.cross(r1, r2)
    R0 = np.stack([r1, r2, r3], axis=1)
    U, _S, Vt = np.linalg.svd(R0)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1.0
        Rm = U @ Vt
    t = (s * h3).reshape(3).astype(np.float64)
    if not np.all(np.isfinite(t)) or t[2] <= 0:
        t = (-t).reshape(3)
        if t[2] <= 0:
            return None
    rvec, _ = cv2.Rodrigues(Rm)
    return rvec.reshape(3).astype(np.float64), t.reshape(3).astype(np.float64)


def _init_coeffs_pinhole_prior(
    *,
    uv_all: np.ndarray,
    nmax: int,
    image_size: tuple[int, int],
    f0_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    from stereocomplex.ray3d.central_ba import default_disk  # noqa: PLC0415
    from stereocomplex.core.model_compact.zernike import zernike_design_matrix  # noqa: PLC0415

    w, h = int(image_size[0]), int(image_size[1])
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    u0, v0, radius = default_disk(w, h)

    uv_all = np.asarray(uv_all, dtype=np.float64).reshape(-1, 2)
    A, mask, _modes = zernike_design_matrix(
        uv_all[:, 0], uv_all[:, 1], nmax=int(nmax), u0_px=float(u0), v0_px=float(v0), radius_px=float(radius)
    )
    if not np.all(mask):
        A = A[mask]
        uv_all = uv_all[mask]

    x_t = (uv_all[:, 0] - cx) / float(f0_px)
    y_t = (uv_all[:, 1] - cy) / float(f0_px)

    lam = 1e-9
    ATA = A.T @ A + lam * np.eye(A.shape[1], dtype=np.float64)
    ax = np.linalg.solve(ATA, A.T @ x_t)
    ay = np.linalg.solve(ATA, A.T @ y_t)
    return ax.astype(np.float64), ay.astype(np.float64)


def _init_coeffs_from_pose_guess(
    *,
    frames: dict[int, FrameObservations],
    rvecs0: dict[int, np.ndarray],
    tvecs0: dict[int, np.ndarray],
    nmax: int,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    import cv2  # type: ignore

    from stereocomplex.ray3d.central_ba import default_disk  # noqa: PLC0415
    from stereocomplex.core.model_compact.zernike import zernike_design_matrix  # noqa: PLC0415

    w, h = int(image_size[0]), int(image_size[1])
    u0, v0, radius = default_disk(w, h)
    uv_all: list[np.ndarray] = []
    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []

    for fid, fr in frames.items():
        rvec = np.asarray(rvecs0[int(fid)], dtype=np.float64).reshape(3)
        tvec = np.asarray(tvecs0[int(fid)], dtype=np.float64).reshape(3)
        rot, _ = cv2.Rodrigues(rvec)
        P_cam = (rot @ fr.P_board_mm.T).T + tvec.reshape(1, 3)
        Z = P_cam[:, 2]
        good = np.isfinite(Z) & (np.abs(Z) > 1e-9)
        if not np.any(good):
            continue
        uv_all.append(np.asarray(fr.uv_px, dtype=np.float64)[good])
        x_all.append((P_cam[good, 0] / Z[good]).astype(np.float64))
        y_all.append((P_cam[good, 1] / Z[good]).astype(np.float64))

    if not uv_all:
        f0_px = 1.5 * float(max(w, h))
        return _init_coeffs_pinhole_prior(uv_all=np.zeros((1, 2)), nmax=int(nmax), image_size=image_size, f0_px=f0_px)

    uv = np.concatenate(uv_all, axis=0)
    x = np.concatenate(x_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    A, mask, _modes = zernike_design_matrix(
        uv[:, 0], uv[:, 1], nmax=int(nmax), u0_px=u0, v0_px=v0, radius_px=radius
    )
    if not np.all(mask):
        A = A[mask]
        x = x[mask]
        y = y[mask]

    lam = 1e-6
    ATA = A.T @ A + lam * np.eye(A.shape[1], dtype=np.float64)
    ax = np.linalg.solve(ATA, A.T @ x)
    ay = np.linalg.solve(ATA, A.T @ y)
    return ax.astype(np.float64), ay.astype(np.float64)


def _rig_from_poses(
    common_fids: list[int],
    rvecs_L: dict[int, np.ndarray],
    tvecs_L: dict[int, np.ndarray],
    rvecs_R: dict[int, np.ndarray],
    tvecs_R: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation as R  # type: ignore

    Rs: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    for fid in common_fids:
        R_L = R.from_rotvec(rvecs_L[fid]).as_matrix()
        R_R = R.from_rotvec(rvecs_R[fid]).as_matrix()
        t_L = tvecs_L[fid].reshape(3)
        t_R = tvecs_R[fid].reshape(3)
        R_RL = R_R @ R_L.T
        t_RL = t_R - R_RL @ t_L
        Rs.append(R_RL)
        ts.append(t_RL)
    rot_mean = R.from_matrix(np.stack(Rs, axis=0)).mean().as_matrix()
    t_mean = np.mean(np.stack(ts, axis=0), axis=0)
    C_R_in_L = -rot_mean.T @ t_mean
    return rot_mean, t_mean, C_R_in_L


def _ray_fit_health_metrics(
    *,
    model: StereoCentralRayFieldModel,
    frames: dict[int, StereoFrameObservations],
    rvecs: dict[int, np.ndarray],
    tvecs: dict[int, np.ndarray],
) -> dict[str, float]:
    from scipy.spatial.transform import Rotation as R  # type: ignore

    skew_values: list[np.ndarray] = []
    residual_values: list[np.ndarray] = []
    n_points_by_frame: list[int] = []

    for fid in sorted(frames):
        fr = frames[int(fid)]
        uv_left = np.asarray(fr.uv_left_px, dtype=np.float64).reshape(-1, 2)
        uv_right = np.asarray(fr.uv_right_px, dtype=np.float64).reshape(-1, 2)
        points_board = np.asarray(fr.P_board_mm, dtype=np.float64).reshape(-1, 3)
        if uv_left.size == 0:
            continue

        rot = R.from_rotvec(np.asarray(rvecs[int(fid)], dtype=np.float64).reshape(3)).as_matrix()
        tvec = np.asarray(tvecs[int(fid)], dtype=np.float64).reshape(3)
        points_left = (rot @ points_board.T).T + tvec.reshape(1, 3)
        points_right = (model.R_RL @ points_left.T).T + model.t_RL.reshape(1, 3)

        dirs_left = model.left.ray_directions_cam(uv_left[:, 0], uv_left[:, 1])
        dirs_right = model.right.ray_directions_cam(uv_right[:, 0], uv_right[:, 1])
        proj_left = np.sum(points_left * dirs_left, axis=-1, keepdims=True) * dirs_left
        proj_right = np.sum(points_right * dirs_right, axis=-1, keepdims=True) * dirs_right
        residual_values.append(np.linalg.norm(points_left - proj_left, axis=-1))
        residual_values.append(np.linalg.norm(points_right - proj_right, axis=-1))

        _xyz, skew = model.triangulate(uv_left, uv_right)
        skew_values.append(np.asarray(skew, dtype=np.float64).reshape(-1))
        n_points_by_frame.append(int(uv_left.shape[0]))

    if not n_points_by_frame:
        raise RuntimeError("cannot compute fit health metrics without training observations")

    skew_all = np.concatenate(skew_values, axis=0)
    residual_all = np.concatenate(residual_values, axis=0)
    return {
        "train_skew_rms_mm": float(np.sqrt(np.mean(skew_all**2))),
        "train_skew_p95_mm": float(np.quantile(skew_all, 0.95)),
        "train_point_to_ray_rms_mm": float(np.sqrt(np.mean(residual_all**2))),
        "train_point_to_ray_p95_mm": float(np.quantile(residual_all, 0.95)),
        "n_points_total": float(sum(n_points_by_frame)),
        "mean_common_corners_per_frame": float(np.mean(n_points_by_frame)),
    }


def _normalize_image_pairs(image_pairs: Sequence[StereoImagePair | tuple[str | Path, str | Path]]) -> list[StereoImagePair]:
    out: list[StereoImagePair] = []
    for k, pair in enumerate(image_pairs):
        if isinstance(pair, StereoImagePair):
            fid = int(pair.frame_id) if pair.frame_id is not None else int(k)
            out.append(StereoImagePair(left_path=Path(pair.left_path), right_path=Path(pair.right_path), frame_id=fid))
            continue
        left_path, right_path = pair
        out.append(StereoImagePair(left_path=Path(left_path), right_path=Path(right_path), frame_id=int(k)))
    return out


def _sorted_image_paths(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in exts)


def fit_opencv_stereo_from_image_pairs(
    *,
    image_pairs: Sequence[StereoImagePair | tuple[str | Path, str | Path]],
    board: CharucoBoardSpec,
    method2d: RefineMethod = "raw",
    min_common_corners: int = 10,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
) -> StereoOpenCVCalibrationResult:
    """
    Calibrate a standard OpenCV stereo pinhole model from image pairs.

    This is the onboarding-friendly companion to the 3D ray-field API: it uses the
    same public ChArUco detection/refinement path, then runs plain OpenCV mono and
    stereo calibration. Use `method2d="raw"` for baseline OpenCV and
    `method2d="rayfield_tps_robust"` for OpenCV fed with refined corners.
    """
    import cv2  # type: ignore

    pairs = _normalize_image_pairs(image_pairs)
    if not pairs:
        raise ValueError("image_pairs is empty")

    runtime = _build_charuco_runtime(board)
    chess3 = np.asarray(runtime.board.getChessboardCorners(), dtype=np.float64)
    image_size: tuple[int, int] | None = None
    detected_pairs = 0
    used_frame_ids: list[int] = []

    obj_left: list[np.ndarray] = []
    img_left: list[np.ndarray] = []
    obj_right: list[np.ndarray] = []
    img_right: list[np.ndarray] = []
    obj_stereo: list[np.ndarray] = []
    img_stereo_left: list[np.ndarray] = []
    img_stereo_right: list[np.ndarray] = []

    for pair in pairs:
        img_left_gray = _ensure_gray_u8(pair.left_path)
        img_right_gray = _ensure_gray_u8(pair.right_path)
        current_size = (int(img_left_gray.shape[1]), int(img_left_gray.shape[0]))
        if image_size is None:
            image_size = current_size
        if current_size != image_size or img_right_gray.shape[:2] != img_left_gray.shape[:2]:
            raise ValueError("all left/right images must share the same size")

        det_left = _detect_view(runtime, img_left_gray)
        det_right = _detect_view(runtime, img_right_gray)
        if det_left is None or det_right is None:
            continue
        detected_pairs += 1

        xy_left = _refine_detection_points(
            runtime=runtime,
            detection=det_left,
            method2d=method2d,
            tps_lam=tps_lam,
            huber_c=huber_c,
            iters=iters,
        )
        xy_right = _refine_detection_points(
            runtime=runtime,
            detection=det_right,
            method2d=method2d,
            tps_lam=tps_lam,
            huber_c=huber_c,
            iters=iters,
        )

        ids_left = np.asarray(det_left.charuco_ids, dtype=np.int32).reshape(-1)
        ids_right = np.asarray(det_right.charuco_ids, dtype=np.int32).reshape(-1)
        if ids_left.size >= 4:
            obj_left.append(np.asarray(chess3[ids_left], dtype=np.float32).reshape(-1, 3))
            img_left.append(np.asarray(xy_left, dtype=np.float32).reshape(-1, 2))
        if ids_right.size >= 4:
            obj_right.append(np.asarray(chess3[ids_right], dtype=np.float32).reshape(-1, 3))
            img_right.append(np.asarray(xy_right, dtype=np.float32).reshape(-1, 2))

        map_left = _dict_from_ids_xy(ids_left, xy_left)
        map_right = _dict_from_ids_xy(ids_right, xy_right)
        common_ids = sorted(set(map_left).intersection(map_right))
        if len(common_ids) < int(min_common_corners):
            continue

        obj_stereo.append(np.asarray(chess3[np.asarray(common_ids, dtype=np.int32)], dtype=np.float32).reshape(-1, 3))
        img_stereo_left.append(np.asarray([map_left[i] for i in common_ids], dtype=np.float32).reshape(-1, 2))
        img_stereo_right.append(np.asarray([map_right[i] for i in common_ids], dtype=np.float32).reshape(-1, 2))
        used_frame_ids.append(int(pair.frame_id if pair.frame_id is not None else len(used_frame_ids)))

    if image_size is None:
        raise RuntimeError("no readable image pair found")
    if len(obj_left) < 2 or len(obj_right) < 2:
        raise RuntimeError("not enough mono calibration frames after ChArUco detection/refinement")
    if len(obj_stereo) < 2:
        raise RuntimeError("not enough stereo calibration frames after ChArUco detection/refinement")

    mono_left_rms, K_left, dist_left, _rvecs_left, _tvecs_left = runtime.cv2.calibrateCamera(
        obj_left,
        img_left,
        image_size,
        None,
        None,
        flags=0,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
    )
    mono_right_rms, K_right, dist_right, _rvecs_right, _tvecs_right = runtime.cv2.calibrateCamera(
        obj_right,
        img_right,
        image_size,
        None,
        None,
        flags=0,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
    )
    stereo_rms, K_left, dist_left, K_right, dist_right, R_lr, t_lr, E, F = runtime.cv2.stereoCalibrate(
        obj_stereo,
        img_stereo_left,
        img_stereo_right,
        K_left,
        dist_left,
        K_right,
        dist_right,
        image_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    report = StereoOpenCVCalibrationReport(
        image_width_px=int(image_size[0]),
        image_height_px=int(image_size[1]),
        n_input_pairs=len(pairs),
        n_detected_pairs=int(detected_pairs),
        n_mono_frames_left=len(obj_left),
        n_mono_frames_right=len(obj_right),
        n_stereo_frames=len(obj_stereo),
        used_frame_ids=tuple(int(fid) for fid in used_frame_ids),
        method2d=method2d,
        min_common_corners=int(min_common_corners),
        mono_left_rms_px=float(mono_left_rms),
        mono_right_rms_px=float(mono_right_rms),
        stereo_rms_px=float(stereo_rms),
        baseline_mm=float(np.linalg.norm(np.asarray(t_lr, dtype=np.float64).reshape(3))),
    )
    return StereoOpenCVCalibrationResult(
        K_left=np.asarray(K_left, dtype=np.float64),
        dist_left=np.asarray(dist_left, dtype=np.float64).reshape(-1),
        K_right=np.asarray(K_right, dtype=np.float64),
        dist_right=np.asarray(dist_right, dtype=np.float64).reshape(-1),
        R_right_from_left=np.asarray(R_lr, dtype=np.float64),
        t_right_from_left_mm=np.asarray(t_lr, dtype=np.float64).reshape(3),
        essential_matrix=np.asarray(E, dtype=np.float64),
        fundamental_matrix=np.asarray(F, dtype=np.float64),
        report=report,
    )


def compare_opencv_stereo_calibration(
    left_dir: str | Path,
    right_dir: str | Path,
    board: CharucoBoardSpec,
    *,
    max_pairs: int | None = None,
    method2d: str = "rayfield_tps_robust",
    **kwargs,
) -> dict:
    """Run OpenCV stereo calibration with raw AND refined corners.

    This is the recommended first step for an OpenCV user: it runs
    ``fit_opencv_stereo_from_image_dirs`` twice (once with
    ``method2d="raw"``, once with *method2d*) and returns a comparison
    dictionary.

    Parameters are forwarded to ``fit_opencv_stereo_from_image_dirs``.
    """
    raw = fit_opencv_stereo_from_image_dirs(
        left_dir=Path(left_dir), right_dir=Path(right_dir),
        board=board, method2d="raw", max_pairs=max_pairs, **kwargs,
    )
    refined = fit_opencv_stereo_from_image_dirs(
        left_dir=Path(left_dir), right_dir=Path(right_dir),
        board=board, method2d=method2d, max_pairs=max_pairs, **kwargs,
    )
    return {
        "raw": raw.to_dict(),
        "refined": refined.to_dict(),
        "raw_result": raw,
        "refined_result": refined,
        "improvement_px": float(raw.report.stereo_rms_px - refined.report.stereo_rms_px),
    }


def fit_opencv_stereo_from_image_dirs(
    *,
    left_dir: str | Path,
    right_dir: str | Path,
    board: CharucoBoardSpec,
    max_pairs: int = 0,
    method2d: RefineMethod = "raw",
    min_common_corners: int = 10,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
) -> StereoOpenCVCalibrationResult:
    left_paths = _sorted_image_paths(Path(left_dir))
    right_paths = _sorted_image_paths(Path(right_dir))
    if not left_paths or not right_paths:
        raise FileNotFoundError("no images found in left_dir/right_dir")
    if len(left_paths) != len(right_paths):
        raise ValueError("left_dir and right_dir must contain the same number of images")
    if max_pairs and max_pairs > 0:
        left_paths = left_paths[: int(max_pairs)]
        right_paths = right_paths[: int(max_pairs)]
    pairs = [
        StereoImagePair(left_path=left_path, right_path=right_path, frame_id=k)
        for k, (left_path, right_path) in enumerate(zip(left_paths, right_paths, strict=True))
    ]
    return fit_opencv_stereo_from_image_pairs(
        image_pairs=pairs,
        board=board,
        method2d=method2d,
        min_common_corners=min_common_corners,
        tps_lam=tps_lam,
        huber_c=huber_c,
        iters=iters,
    )


def fit_opencv_stereo_from_dataset(
    *,
    dataset_root: str | Path,
    split: str = "train",
    scene: str = "scene_0000",
    max_frames: int = 0,
    method2d: RefineMethod = "raw",
    min_common_corners: int = 10,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
) -> StereoOpenCVCalibrationResult:
    dataset_root = Path(dataset_root)
    scene_dir = dataset_root / str(split) / str(scene)
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
    return fit_opencv_stereo_from_image_pairs(
        image_pairs=pairs,
        board=board,
        method2d=method2d,
        min_common_corners=min_common_corners,
        tps_lam=tps_lam,
        huber_c=huber_c,
        iters=iters,
    )


def fit_stereo_central_rayfield_from_image_pairs(
    *,
    image_pairs: Sequence[StereoImagePair | tuple[str | Path, str | Path]],
    board: CharucoBoardSpec,
    method2d: RefineMethod = "rayfield_tps_robust",
    min_common_corners: int = 10,
    nmax: int = 10,
    lam_coeff: float = 1e-3,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
    max_nfev: int = 800,
    export_model_dir: str | Path | None = None,
) -> StereoCentralRayFieldFitResult:
    """
    Calibrate a public stereo central ray-field model directly from image pairs.

    This is the stable high-level entry point for users who already have left/right
    image pairs and a known ChArUco board.
    """
    pairs = _normalize_image_pairs(image_pairs)
    if not pairs:
        raise ValueError("image_pairs is empty")

    runtime = _build_charuco_runtime(board)
    chess3 = np.asarray(runtime.board.getChessboardCorners(), dtype=np.float64)
    obs_by_side: dict[str, dict[int, FrameObservations]] = {"left": {}, "right": {}}
    detected_pairs = 0
    image_size: tuple[int, int] | None = None

    for pair in pairs:
        img_left = _ensure_gray_u8(pair.left_path)
        img_right = _ensure_gray_u8(pair.right_path)
        if image_size is None:
            image_size = (int(img_left.shape[1]), int(img_left.shape[0]))
        if img_left.shape != img_right.shape:
            raise ValueError(f"left/right image size mismatch for frame {pair.frame_id}")
        if image_size != (int(img_left.shape[1]), int(img_left.shape[0])):
            raise ValueError("all images must have the same size")

        det_left = _detect_view(runtime, img_left)
        det_right = _detect_view(runtime, img_right)
        if det_left is None or det_right is None:
            continue
        detected_pairs += 1

        xy_left = _refine_detection_points(
            runtime=runtime,
            detection=det_left,
            method2d=method2d,
            tps_lam=tps_lam,
            huber_c=huber_c,
            iters=iters,
        )
        xy_right = _refine_detection_points(
            runtime=runtime,
            detection=det_right,
            method2d=method2d,
            tps_lam=tps_lam,
            huber_c=huber_c,
            iters=iters,
        )
        map_left = _dict_from_ids_xy(det_left.charuco_ids, xy_left)
        map_right = _dict_from_ids_xy(det_right.charuco_ids, xy_right)

        common_ids = sorted(set(map_left).intersection(map_right))
        if len(common_ids) < int(min_common_corners):
            continue

        uv_left = np.stack([map_left[i] for i in common_ids], axis=0).astype(np.float64)
        uv_right = np.stack([map_right[i] for i in common_ids], axis=0).astype(np.float64)
        obj = chess3[np.asarray(common_ids, dtype=np.int32)].astype(np.float64)
        fid = int(pair.frame_id if pair.frame_id is not None else len(obs_by_side["left"]))
        obs_by_side["left"][fid] = FrameObservations(uv_px=uv_left, P_board_mm=obj)
        obs_by_side["right"][fid] = FrameObservations(uv_px=uv_right, P_board_mm=obj)

    if not obs_by_side["left"] or not obs_by_side["right"]:
        raise RuntimeError("no usable stereo frames after ChArUco detection/refinement")
    assert image_size is not None

    def _init_poses(side: str) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        homographies: list[np.ndarray] = []
        for fr_obs in obs_by_side[side].values():
            obj_xy = np.asarray(fr_obs.P_board_mm, dtype=np.float64)[:, :2]
            uv = np.asarray(fr_obs.uv_px, dtype=np.float64)
            if obj_xy.shape[0] < 6:
                continue
            H, _mask = runtime.cv2.findHomography(
                obj_xy, uv, method=runtime.cv2.RANSAC, ransacReprojThreshold=3.0
            )
            if H is None:
                continue
            homographies.append(np.asarray(H, dtype=np.float64))
        K0 = _estimate_K0_from_homographies(homographies=homographies, image_size=image_size)

        rvecs0: dict[int, np.ndarray] = {}
        tvecs0: dict[int, np.ndarray] = {}
        for fid, fr_obs in obs_by_side[side].items():
            obj_xy = np.asarray(fr_obs.P_board_mm, dtype=np.float64)[:, :2]
            uv = np.asarray(fr_obs.uv_px, dtype=np.float64)
            init = _init_pose_from_homography(runtime.cv2, obj_xy_mm=obj_xy, uv_px=uv, K0=K0)
            if init is None:
                continue
            rvecs0[fid], tvecs0[fid] = init
        return rvecs0, tvecs0

    rvecs0_L, tvecs0_L = _init_poses("left")
    rvecs0_R, tvecs0_R = _init_poses("right")
    common = sorted(set(rvecs0_L).intersection(rvecs0_R).intersection(obs_by_side["left"]).intersection(obs_by_side["right"]))
    if len(common) < 2:
        raise RuntimeError("not enough frames with initialized poses")

    obs_left = {fid: obs_by_side["left"][fid] for fid in common}
    obs_right = {fid: obs_by_side["right"][fid] for fid in common}
    rvecs0_L = {fid: rvecs0_L[fid] for fid in common}
    tvecs0_L = {fid: tvecs0_L[fid] for fid in common}
    rvecs0_R = {fid: rvecs0_R[fid] for fid in common}
    tvecs0_R = {fid: tvecs0_R[fid] for fid in common}

    coeffs0_x_L, coeffs0_y_L = _init_coeffs_from_pose_guess(
        frames=obs_left,
        rvecs0=rvecs0_L,
        tvecs0=tvecs0_L,
        nmax=int(nmax),
        image_size=image_size,
    )
    coeffs0_x_R, coeffs0_y_R = _init_coeffs_from_pose_guess(
        frames=obs_right,
        rvecs0=rvecs0_R,
        tvecs0=tvecs0_R,
        nmax=int(nmax),
        image_size=image_size,
    )

    stereo_frames = {
        fid: StereoFrameObservations(
            uv_left_px=obs_left[fid].uv_px,
            uv_right_px=obs_right[fid].uv_px,
            P_board_mm=obs_left[fid].P_board_mm,
        )
        for fid in common
    }
    R_RL0, t_RL0, _C_R_in_L0 = _rig_from_poses(common, rvecs0_L, tvecs0_L, rvecs0_R, tvecs0_R)
    rig_rvec0, _ = runtime.cv2.Rodrigues(R_RL0)
    res = fit_central_stereo_rayfield_ba(
        frames=stereo_frames,
        image_width_px=int(image_size[0]),
        image_height_px=int(image_size[1]),
        nmax=int(nmax),
        rvecs0=rvecs0_L,
        tvecs0=tvecs0_L,
        rig_rvec0=rig_rvec0.reshape(3),
        rig_tvec0=t_RL0.reshape(3),
        coeffs0_left_x=coeffs0_x_L,
        coeffs0_left_y=coeffs0_y_L,
        coeffs0_right_x=coeffs0_x_R,
        coeffs0_right_y=coeffs0_y_R,
        lam_coeff=float(lam_coeff),
        lam_center=1e-1,
        lam_jacobian=10.0,
        loss="huber",
        f_scale_mm=1.0,
        max_nfev=int(max_nfev),
    )

    from scipy.spatial.transform import Rotation as R  # type: ignore

    R_RL = R.from_rotvec(res.rig_rvec.reshape(3)).as_matrix()
    t_RL = res.rig_tvec.reshape(3)
    model = StereoCentralRayFieldModel.from_coeffs(
        image_width_px=int(image_size[0]),
        image_height_px=int(image_size[1]),
        nmax=int(res.nmax),
        u0_px=float(res.u0_px),
        v0_px=float(res.v0_px),
        radius_px=float(res.radius_px),
        coeffs_left_x=np.asarray(res.coeffs_left_x, dtype=np.float64),
        coeffs_left_y=np.asarray(res.coeffs_left_y, dtype=np.float64),
        coeffs_right_x=np.asarray(res.coeffs_right_x, dtype=np.float64),
        coeffs_right_y=np.asarray(res.coeffs_right_y, dtype=np.float64),
        R_RL=np.asarray(R_RL, dtype=np.float64),
        t_RL=np.asarray(t_RL, dtype=np.float64).reshape(3),
    )
    fit_health = _ray_fit_health_metrics(
        model=model,
        frames=stereo_frames,
        rvecs=res.rvecs,
        tvecs=res.tvecs,
    )

    exported_model_json = None
    if export_model_dir is not None:
        exported_model_json = str(save_stereo_central_rayfield(Path(export_model_dir), model))

    report = StereoCentralRayFieldFitReport(
        image_width_px=int(image_size[0]),
        image_height_px=int(image_size[1]),
        n_input_pairs=len(pairs),
        n_detected_pairs=int(detected_pairs),
        n_observation_frames=len(obs_by_side["left"]),
        n_initialized_frames=len(common),
        used_frame_ids=tuple(int(fid) for fid in common),
        method2d=method2d,
        min_common_corners=int(min_common_corners),
        nmax=int(nmax),
        train_skew_rms_mm=float(fit_health["train_skew_rms_mm"]),
        train_skew_p95_mm=float(fit_health["train_skew_p95_mm"]),
        train_point_to_ray_rms_mm=float(fit_health["train_point_to_ray_rms_mm"]),
        train_point_to_ray_p95_mm=float(fit_health["train_point_to_ray_p95_mm"]),
        n_points_total=int(fit_health["n_points_total"]),
        mean_common_corners_per_frame=float(fit_health["mean_common_corners_per_frame"]),
        diagnostics={**dict(res.diagnostics), **fit_health},
        exported_model_json=exported_model_json,
    )
    return StereoCentralRayFieldFitResult(model=model, report=report)


def fit_stereo_central_rayfield_from_image_dirs(
    *,
    left_dir: str | Path,
    right_dir: str | Path,
    board: CharucoBoardSpec,
    max_pairs: int = 0,
    method2d: RefineMethod = "rayfield_tps_robust",
    min_common_corners: int = 10,
    nmax: int = 10,
    lam_coeff: float = 1e-3,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
    max_nfev: int = 800,
    export_model_dir: str | Path | None = None,
) -> StereoCentralRayFieldFitResult:
    left_paths = _sorted_image_paths(Path(left_dir))
    right_paths = _sorted_image_paths(Path(right_dir))
    if not left_paths or not right_paths:
        raise FileNotFoundError("no images found in left_dir/right_dir")
    if len(left_paths) != len(right_paths):
        raise ValueError("left_dir and right_dir must contain the same number of images")
    if max_pairs and max_pairs > 0:
        left_paths = left_paths[: int(max_pairs)]
        right_paths = right_paths[: int(max_pairs)]
    pairs = [
        StereoImagePair(left_path=left_path, right_path=right_path, frame_id=k)
        for k, (left_path, right_path) in enumerate(zip(left_paths, right_paths, strict=True))
    ]
    return fit_stereo_central_rayfield_from_image_pairs(
        image_pairs=pairs,
        board=board,
        method2d=method2d,
        min_common_corners=min_common_corners,
        nmax=nmax,
        lam_coeff=lam_coeff,
        tps_lam=tps_lam,
        huber_c=huber_c,
        iters=iters,
        max_nfev=max_nfev,
        export_model_dir=export_model_dir,
    )


def fit_stereo_zernike_origin_field_from_image_dirs(
    *,
    left_dir: str | Path,
    right_dir: str | Path,
    board: CharucoBoardSpec,
    max_order: int = 4,
    max_pairs: int = 0,
    method2d: RefineMethod = "raw",
    min_common_corners: int = 10,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
    regularization: float = 1e-6,
    max_nfev: int = 200,
):
    """
    Calibrate a non-central stereo Zernike origin-field model from image directories.

    Detects ChArUco corners, runs OpenCV mono+stereo calibration to obtain K and
    the stereo rig transform, solves per-frame board poses, then fits a Zernike
    origin field O(u,v) on top of the pinhole directions.

    Parameters
    ----------
    left_dir, right_dir:
        Directories containing sorted left/right calibration images (same count).
    board:
        ChArUco board specification.
    max_order:
        Maximum Zernike radial order for the origin field.
    max_pairs:
        Cap on the number of image pairs used (0 = use all).
    method2d:
        Corner refinement method passed to refine_charuco_corners.
    min_common_corners:
        Minimum corners common to left and right per frame.
    regularization:
        L2 regularization weight on the Zernike origin coefficients.
    max_nfev:
        Maximum function evaluations for the Zernike BA.

    Returns
    -------
    StereoZernikeOriginFieldFitResult

    Notes
    -----
    All frames are restricted to the intersection of corners visible in EVERY
    frame, because `SyntheticStereoDataset` requires a single shared
    `object_points` array.  Corners not detected in all frames are dropped.
    If this intersection is too small, add more frames with full board coverage
    or reduce `min_common_corners`.
    """
    import cv2  # type: ignore

    from stereocomplex.calibration.fit_zernike_origin_field import fit_stereo_zernike_origin_field  # noqa: PLC0415
    from stereocomplex.rayfields.zernike_origin_field import ZernikeOriginFieldConfig  # noqa: PLC0415
    from stereocomplex.synthetic.parallel_plate import SyntheticStereoDataset  # noqa: PLC0415

    left_paths = _sorted_image_paths(Path(left_dir))
    right_paths = _sorted_image_paths(Path(right_dir))
    if not left_paths or not right_paths:
        raise FileNotFoundError("no images found in left_dir/right_dir")
    if len(left_paths) != len(right_paths):
        raise ValueError("left_dir and right_dir must contain the same number of images")
    if max_pairs and max_pairs > 0:
        left_paths = left_paths[: int(max_pairs)]
        right_paths = right_paths[: int(max_pairs)]

    runtime = _build_charuco_runtime(board)
    chess3 = np.asarray(runtime.board.getChessboardCorners(), dtype=np.float64)
    image_size: tuple[int, int] | None = None

    frame_maps_left: list[dict[int, np.ndarray]] = []
    frame_maps_right: list[dict[int, np.ndarray]] = []
    frame_common_ids: list[list[int]] = []
    obj_left: list[np.ndarray] = []
    img_left_cv: list[np.ndarray] = []
    obj_right: list[np.ndarray] = []
    img_right_cv: list[np.ndarray] = []

    for left_path, right_path in zip(left_paths, right_paths, strict=True):
        img_l = _ensure_gray_u8(left_path)
        img_r = _ensure_gray_u8(right_path)
        if image_size is None:
            image_size = (int(img_l.shape[1]), int(img_l.shape[0]))
        if img_l.shape != img_r.shape or (int(img_l.shape[1]), int(img_l.shape[0])) != image_size:
            continue

        det_l = _detect_view(runtime, img_l)
        det_r = _detect_view(runtime, img_r)
        if det_l is None or det_r is None:
            continue

        xy_l = _refine_detection_points(
            runtime=runtime, detection=det_l, method2d=method2d,
            tps_lam=tps_lam, huber_c=huber_c, iters=iters,
        )
        xy_r = _refine_detection_points(
            runtime=runtime, detection=det_r, method2d=method2d,
            tps_lam=tps_lam, huber_c=huber_c, iters=iters,
        )
        ids_l = np.asarray(det_l.charuco_ids, dtype=np.int32).reshape(-1)
        ids_r = np.asarray(det_r.charuco_ids, dtype=np.int32).reshape(-1)
        map_l = _dict_from_ids_xy(ids_l, xy_l)
        map_r = _dict_from_ids_xy(ids_r, xy_r)
        common_ids = sorted(set(map_l).intersection(map_r))
        if len(common_ids) < int(min_common_corners):
            continue

        frame_maps_left.append(map_l)
        frame_maps_right.append(map_r)
        frame_common_ids.append(common_ids)

        if ids_l.size >= 4:
            obj_left.append(chess3[ids_l].astype(np.float32).reshape(-1, 3))
            img_left_cv.append(xy_l.astype(np.float32).reshape(-1, 2))
        if ids_r.size >= 4:
            obj_right.append(chess3[ids_r].astype(np.float32).reshape(-1, 3))
            img_right_cv.append(xy_r.astype(np.float32).reshape(-1, 2))

    if not frame_common_ids:
        raise RuntimeError("no usable stereo frames after ChArUco detection")
    assert image_size is not None
    if len(obj_left) < 2 or len(obj_right) < 2:
        raise RuntimeError("not enough frames for mono calibration (need ≥ 2 per side)")

    _, K_left_cv, dist_left_cv, _, _ = cv2.calibrateCamera(
        obj_left, img_left_cv, image_size, None, None, flags=0,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
    )
    _, K_right_cv, dist_right_cv, _, _ = cv2.calibrateCamera(
        obj_right, img_right_cv, image_size, None, None, flags=0,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
    )

    obj_stereo = [
        chess3[np.asarray(cids, dtype=np.int32)].astype(np.float32).reshape(-1, 3)
        for cids in frame_common_ids
    ]
    img_stereo_l = [
        np.stack([frame_maps_left[i][c] for c in frame_common_ids[i]], axis=0).astype(np.float32)
        for i in range(len(frame_common_ids))
    ]
    img_stereo_r = [
        np.stack([frame_maps_right[i][c] for c in frame_common_ids[i]], axis=0).astype(np.float32)
        for i in range(len(frame_common_ids))
    ]

    _, K_left_cv, dist_left_cv, K_right_cv, dist_right_cv, R_rl, t_rl, _, _ = cv2.stereoCalibrate(
        obj_stereo, img_stereo_l, img_stereo_r,
        K_left_cv, dist_left_cv, K_right_cv, dist_right_cv,
        image_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    K_left = np.asarray(K_left_cv, dtype=np.float64)
    K_right = np.asarray(K_right_cv, dtype=np.float64)
    # Distortion coefficients seed solvePnP but are intentionally not forwarded to
    # the Zernike BA: the origin field O(u,v) absorbs non-pinhole behaviour directly.
    dist_left = np.asarray(dist_left_cv, dtype=np.float64).reshape(-1)
    R_rl_arr = np.asarray(R_rl, dtype=np.float64)
    t_rl_arr = np.asarray(t_rl, dtype=np.float64).reshape(3)

    T_right_left = np.eye(4, dtype=np.float64)
    T_right_left[:3, :3] = R_rl_arr
    T_right_left[:3, 3] = t_rl_arr

    # SyntheticStereoDataset requires a single shared object_points array, so we
    # restrict to corners visible in EVERY frame.  Frames with partial board
    # coverage are restricted to this global intersection; corners unique to a
    # subset of frames are dropped.
    global_common: set[int] = set(frame_common_ids[0])
    for cids in frame_common_ids[1:]:
        global_common = global_common.intersection(cids)
    global_ids = sorted(global_common)
    if len(global_ids) < int(min_common_corners):
        raise RuntimeError(
            f"only {len(global_ids)} corners visible in every frame; need {min_common_corners}. "
            "Try reducing min_common_corners or adding more frames with full board coverage."
        )

    object_points_3d = chess3[np.asarray(global_ids, dtype=np.int32)].astype(np.float64)
    board_poses: list[np.ndarray] = []
    left_pixels: list[np.ndarray] = []
    right_pixels: list[np.ndarray] = []

    for i in range(len(frame_common_ids)):
        uv_l = np.stack([frame_maps_left[i][c] for c in global_ids], axis=0).astype(np.float64)
        uv_r = np.stack([frame_maps_right[i][c] for c in global_ids], axis=0).astype(np.float64)
        obj_pts_f = object_points_3d.astype(np.float32).reshape(-1, 3)
        success, rvec, tvec = cv2.solvePnP(
            obj_pts_f,
            uv_l.astype(np.float32).reshape(-1, 2),
            K_left.astype(np.float32),
            dist_left.astype(np.float32),
        )
        if not success:
            continue
        R_board, _ = cv2.Rodrigues(rvec)
        T_board = np.eye(4, dtype=np.float64)
        T_board[:3, :3] = np.asarray(R_board, dtype=np.float64)
        T_board[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
        board_poses.append(T_board)
        left_pixels.append(uv_l)
        right_pixels.append(uv_r)

    if not board_poses:
        raise RuntimeError("solvePnP failed for all frames")

    dataset = SyntheticStereoDataset(
        object_points=object_points_3d,
        board_poses=board_poses,
        left_pixels=left_pixels,
        right_pixels=right_pixels,
        K_left=K_left,
        K_right=K_right,
        T_left_world=np.eye(4, dtype=np.float64),
        T_right_world=T_right_left.copy(),
        image_size=image_size,
        oracle_left_params=None,
        oracle_right_params=None,
    )

    config = ZernikeOriginFieldConfig(image_size=image_size, max_order=int(max_order))
    return fit_stereo_zernike_origin_field(
        observations=dataset,
        K_left=K_left,
        K_right=K_right,
        T_right_left_initial=T_right_left,
        board_poses_initial=board_poses,
        config_left=config,
        config_right=config,
        regularization=float(regularization),
        max_nfev=int(max_nfev),
    )


def fit_stereo_central_rayfield_from_dataset(
    *,
    dataset_root: str | Path,
    split: str = "train",
    scene: str = "scene_0000",
    max_frames: int = 0,
    method2d: RefineMethod = "rayfield_tps_robust",
    min_common_corners: int = 10,
    nmax: int = 10,
    lam_coeff: float = 1e-3,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
    max_nfev: int = 800,
    export_model_dir: str | Path | None = None,
) -> StereoCentralRayFieldFitResult:
    dataset_root = Path(dataset_root)
    scene_dir = dataset_root / str(split) / str(scene)
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
    return fit_stereo_central_rayfield_from_image_pairs(
        image_pairs=pairs,
        board=board,
        method2d=method2d,
        min_common_corners=min_common_corners,
        nmax=nmax,
        lam_coeff=lam_coeff,
        tps_lam=tps_lam,
        huber_c=huber_c,
        iters=iters,
        max_nfev=max_nfev,
        export_model_dir=export_model_dir,
    )
