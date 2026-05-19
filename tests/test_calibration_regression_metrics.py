from __future__ import annotations
import pytest

import json
from pathlib import Path

import numpy as np


# These limits are intentionally broad: they catch broken plumbing or severe numerical
# regressions without freezing the current optimizer to an over-specific solution.
RAW_STEREO_RMS_MAX = 5.0
REFINED_STEREO_RMS_MAX = 5.0
TRAIN_SKEW_P95_MAX_MM = 100.0
TRAIN_POINT_TO_RAY_P95_MAX_MM = 100.0
MIN_POINTS_TOTAL = 20


def _sample_scene() -> tuple[Path, object]:
    import stereocomplex as sc

    scene_dir = Path("dataset/v0_png/train/scene_0000")
    meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
    return scene_dir, sc.CharucoBoardSpec.from_meta(meta)


def test_opencv_raw_and_refined_regression_metrics() -> None:
    import stereocomplex as sc

    scene_dir, board = _sample_scene()
    raw = sc.fit_opencv_stereo_from_image_dirs(
        left_dir=scene_dir / "left",
        right_dir=scene_dir / "right",
        board=board,
        max_pairs=3,
        method2d="raw",
    )
    refined = sc.fit_opencv_stereo_from_image_dirs(
        left_dir=scene_dir / "left",
        right_dir=scene_dir / "right",
        board=board,
        max_pairs=3,
        method2d="rayfield_tps_robust",
    )

    assert raw.report.n_stereo_frames >= 2
    assert refined.report.n_stereo_frames >= 2
    assert raw.report.stereo_rms_px < RAW_STEREO_RMS_MAX
    assert refined.report.stereo_rms_px < REFINED_STEREO_RMS_MAX
    assert np.isfinite(raw.report.baseline_mm)
    assert np.isfinite(refined.report.baseline_mm)


@pytest.mark.slow
def test_rayfield3d_fit_health_and_reload_regression_metrics(tmp_path: Path) -> None:
    import stereocomplex as sc

    scene_dir, board = _sample_scene()
    result = sc.fit_stereo_central_rayfield_from_image_dirs(
        left_dir=scene_dir / "left",
        right_dir=scene_dir / "right",
        board=board,
        max_pairs=3,
        method2d="rayfield_tps_robust",
        nmax=4,
        max_nfev=80,
        export_model_dir=tmp_path / "rayfield3d",
    )
    report = result.report

    assert report.n_initialized_frames >= 2
    assert report.n_points_total >= MIN_POINTS_TOTAL
    assert report.mean_common_corners_per_frame >= report.min_common_corners
    assert np.isfinite(report.train_skew_rms_mm)
    assert np.isfinite(report.train_skew_p95_mm)
    assert np.isfinite(report.train_point_to_ray_rms_mm)
    assert np.isfinite(report.train_point_to_ray_p95_mm)
    assert report.train_skew_p95_mm < TRAIN_SKEW_P95_MAX_MM
    assert report.train_point_to_ray_p95_mm < TRAIN_POINT_TO_RAY_P95_MAX_MM

    reloaded = sc.load_stereo_central_rayfield(tmp_path / "rayfield3d")
    detection_left = sc.detect_charuco_corners(image=scene_dir / "left" / "000000.png", board=board)
    detection_right = sc.detect_charuco_corners(image=scene_dir / "right" / "000000.png", board=board)
    assert detection_left is not None
    assert detection_right is not None
    ids_left = {int(cid): idx for idx, cid in enumerate(detection_left.charuco_ids.tolist())}
    ids_right = {int(cid): idx for idx, cid in enumerate(detection_right.charuco_ids.tolist())}
    common_ids = sorted(set(ids_left).intersection(ids_right))[:10]
    assert common_ids
    uv_left = np.stack([detection_left.charuco_xy[ids_left[cid]] for cid in common_ids], axis=0)
    uv_right = np.stack([detection_right.charuco_xy[ids_right[cid]] for cid in common_ids], axis=0)
    xyz, skew = reloaded.triangulate(uv_left, uv_right)
    assert xyz.shape == (len(common_ids), 3)
    assert skew.shape == (len(common_ids),)
    assert np.all(np.isfinite(xyz))
    assert np.all(np.isfinite(skew))
