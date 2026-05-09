from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_fit_stereo_central_rayfield_from_dataset_smoke(tmp_path: Path) -> None:
    from stereocomplex.advanced import fit_stereo_central_rayfield_from_dataset
    import stereocomplex as sc

    scene_root = Path("dataset/v0_png")
    result = fit_stereo_central_rayfield_from_dataset(
        dataset_root=scene_root,
        split="train",
        scene="scene_0000",
        max_frames=3,
        method2d="rayfield_tps_robust",
        nmax=4,
        max_nfev=80,
        export_model_dir=tmp_path / "model_from_dataset",
    )

    assert result.report.n_initialized_frames >= 2
    assert result.report.n_points_total >= 20
    assert np.isfinite(result.report.train_skew_p95_mm)
    assert np.isfinite(result.report.train_point_to_ray_p95_mm)
    assert result.report.exported_model_json is not None

    reloaded = sc.load_stereo_central_rayfield(tmp_path / "model_from_dataset")
    XYZ, skew = reloaded.triangulate(
        np.array([[320.0, 240.0]], dtype=np.float64),
        np.array([[318.0, 240.0]], dtype=np.float64),
    )
    assert XYZ.shape == (1, 3)
    assert skew.shape == (1,)
    assert np.all(np.isfinite(XYZ))
    assert np.all(np.isfinite(skew))


def test_fit_stereo_central_rayfield_from_image_dirs_smoke(tmp_path: Path) -> None:
    import stereocomplex as sc

    scene_dir = Path("dataset/v0_png/train/scene_0000")
    board = sc.CharucoBoardSpec.from_meta(json.loads((scene_dir / "meta.json").read_text(encoding="utf-8")))
    result = sc.fit_stereo_central_rayfield_from_image_dirs(
        left_dir=scene_dir / "left",
        right_dir=scene_dir / "right",
        board=board,
        max_pairs=3,
        method2d="rayfield_tps_robust",
        nmax=4,
        max_nfev=80,
        export_model_dir=tmp_path / "model_from_dirs",
    )

    assert result.report.n_initialized_frames >= 2
    assert result.report.n_points_total >= 20
    assert np.isfinite(result.report.train_skew_p95_mm)
    assert np.isfinite(result.report.train_point_to_ray_p95_mm)
    assert result.report.exported_model_json is not None


def test_fit_opencv_stereo_from_image_dirs_smoke() -> None:
    import stereocomplex as sc

    scene_dir = Path("dataset/v0_png/train/scene_0000")
    board = sc.CharucoBoardSpec.from_meta(json.loads((scene_dir / "meta.json").read_text(encoding="utf-8")))

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
    assert raw.K_left.shape == (3, 3)
    assert refined.K_right.shape == (3, 3)
    assert raw.t_right_from_left_mm.shape == (3,)
    assert refined.t_right_from_left_mm.shape == (3,)
    assert np.isfinite(raw.report.stereo_rms_px)
    assert np.isfinite(refined.report.stereo_rms_px)
