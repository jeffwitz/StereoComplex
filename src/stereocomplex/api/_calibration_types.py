from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stereocomplex.api.corner_refinement import CharucoDetections, RefineMethod
from stereocomplex.api.stereo_reconstruction import StereoCentralRayFieldModel


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
    legacy_pattern: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CharucoBoardSpec:
        """Build a CharucoBoardSpec from a configuration dictionary.

        Parameters
        ----------
        payload : dict
            Dictionary with board geometry fields.

        Returns
        -------
        CharucoBoardSpec
        """
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
            corner_refinement_max_iterations=int(
                payload.get("corner_refinement_max_iterations", 50)
            ),
            corner_refinement_min_accuracy=float(
                payload.get("corner_refinement_min_accuracy", 1e-3)
            ),
            check_markers=bool(payload["check_markers"])
            if payload.get("check_markers") is not None
            else None,
            min_markers=int(payload["min_markers"])
            if payload.get("min_markers") is not None
            else None,
            try_refine_markers=(
                bool(payload["try_refine_markers"])
                if payload.get("try_refine_markers") is not None
                else None
            ),
            legacy_pattern=bool(payload.get("legacy_pattern", False)),
        )

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> CharucoBoardSpec:
        """Build a CharucoBoardSpec from dataset metadata.

        Parameters
        ----------
        meta : dict
            Dataset metadata dictionary with a 'board' key.

        Returns
        -------
        CharucoBoardSpec
        """
        board_meta = meta["board"]
        opencv_meta = meta.get("opencv", {})
        opencv_aruco = (
            opencv_meta.get("aruco_detector", {}) if isinstance(opencv_meta, dict) else {}
        )
        opencv_charuco = (
            opencv_meta.get("charuco_detector", {}) if isinstance(opencv_meta, dict) else {}
        )
        return cls(
            squares_x=int(board_meta["squares_x"]),
            squares_y=int(board_meta["squares_y"]),
            square_size_mm=float(board_meta["square_size_mm"]),
            marker_size_mm=float(board_meta["marker_size_mm"]),
            aruco_dictionary=str(board_meta.get("aruco_dictionary", "DICT_4X4_1000")),
            adaptive_thresh_win_size_max=(
                int(opencv_aruco["adaptiveThreshWinSizeMax"])
                if isinstance(opencv_aruco, dict)
                and opencv_aruco.get("adaptiveThreshWinSizeMax") is not None
                else None
            ),
            corner_refinement_win_size=int(opencv_aruco.get("cornerRefinementWinSize", 5))
            if isinstance(opencv_aruco, dict)
            else 5,
            corner_refinement_max_iterations=int(
                opencv_aruco.get("cornerRefinementMaxIterations", 50)
            )
            if isinstance(opencv_aruco, dict)
            else 50,
            corner_refinement_min_accuracy=float(
                opencv_aruco.get("cornerRefinementMinAccuracy", 1e-3)
            )
            if isinstance(opencv_aruco, dict)
            else 1e-3,
            check_markers=(
                bool(opencv_charuco["checkMarkers"])
                if isinstance(opencv_charuco, dict)
                and opencv_charuco.get("checkMarkers") is not None
                else None
            ),
            min_markers=(
                int(opencv_charuco["minMarkers"])
                if isinstance(opencv_charuco, dict) and opencv_charuco.get("minMarkers") is not None
                else None
            ),
            try_refine_markers=(
                bool(opencv_charuco["tryRefineMarkers"])
                if isinstance(opencv_charuco, dict)
                and opencv_charuco.get("tryRefineMarkers") is not None
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
class CameraSetup:
    name: str
    image_dir: Path


@dataclass(frozen=True)
class NCameraCalibrationResult:
    """Public calibration result for named camera sets."""

    channel_names: tuple[str, ...]
    stereo_result: StereoCentralRayFieldFitResult | None = None

    @property
    def n_channels(self) -> int:
        """Number of channels in this multi-camera dataset."""
        return len(self.channel_names)


@dataclass(frozen=True)
class _RefinedStereoDetections:
    det_left: CharucoDetections
    det_right: CharucoDetections
    xy_left: np.ndarray
    xy_right: np.ndarray
    map_left: dict[int, np.ndarray]
    map_right: dict[int, np.ndarray]


@dataclass(frozen=True)
class _OriginFieldImageObservations:
    image_size: tuple[int, int]
    frame_maps_left: list[dict[int, np.ndarray]]
    frame_maps_right: list[dict[int, np.ndarray]]
    frame_common_ids: list[list[int]]
    obj_left: list[np.ndarray]
    img_left_cv: list[np.ndarray]
    obj_right: list[np.ndarray]
    img_right_cv: list[np.ndarray]


@dataclass(frozen=True)
class _OriginFieldPinholeSeed:
    K_left: np.ndarray
    K_right: np.ndarray
    dist_left: np.ndarray
    T_right_left: np.ndarray


@dataclass(frozen=True)
class _OriginFieldDatasetSeed:
    dataset: Any
    board_poses: list[np.ndarray]


@dataclass(frozen=True)
class StereoCentralRayFieldFitReport:
    """Quality metrics for a fitted stereo central rayfield calibration."""

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
    """Fitted stereo central rayfield model plus calibration report."""

    model: StereoCentralRayFieldModel
    report: StereoCentralRayFieldFitReport


@dataclass(frozen=True)
class StereoOpenCVCalibrationReport:
    """Frame counts and RMS metrics for OpenCV stereo calibration."""

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
    """OpenCV stereo intrinsics, distortion, rig transform and report."""

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
            "mono_rms_left_px": float(self.report.mono_left_rms_px),
            "mono_rms_right_px": float(self.report.mono_right_rms_px),
            "n_stereo_frames": int(self.report.n_stereo_frames),
        }

    def to_opencv(self) -> tuple:
        """Return ``(K1, d1, K2, d2, R, T)`` as used by OpenCV."""
        return (
            self.K_left,
            self.dist_left,
            self.K_right,
            self.dist_right,
            self.R_right_from_left,
            self.t_right_from_left_mm,
        )
