from __future__ import annotations

from pathlib import Path

import pytest

from stereocomplex.sim.dataset_validate import validate_dataset


@pytest.mark.integration
def test_pipeline_generate_validate_refine(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco"):
        pytest.skip("cv2.aruco not available (need opencv-contrib-python)")

    from stereocomplex.sim.cpu.generate_dataset import generate_cpu_dataset
    from stereocomplex.cli.refine_corners import refine_dataset_scene

    out_root = tmp_path / "ds"
    generate_cpu_dataset(
        out_root=out_root,
        scenes=1,
        frames_per_scene=2,
        width=160,
        height=120,
        pattern="charuco",
        tex_interp="linear",
        distort="brown",
        distort_strength=0.2,
        image_format="webp",
        outside_mask="hard",
        blur_fwhm_um=0.0,
        blur_fwhm_px=0.0,
        blur_edge_factor=1.0,
        blur_edge_start=0.6,
        blur_edge_power=2.0,
        noise_std=0.01,
        seed=123,
        pitch_um_override=None,
        f_um_override=None,
        tz_nominal_mm_override=None,
        baseline_mm_override=None,
        board_squares_x_override=None,
        board_squares_y_override=None,
        board_square_size_mm_override=None,
        board_marker_size_mm_override=None,
        board_pixels_per_square_override=None,
    )

    validate_dataset(out_root)

    refined = refine_dataset_scene(
        dataset_root=out_root,
        split="train",
        scene="scene_0000",
        method="rayfield_tps_robust",
        max_frames=2,
        tps_lam=10.0,
        huber_c=3.0,
        iters=2,
    )
    assert refined["schema_version"] == "stereocomplex.refined_corners.v0"
    assert len(refined["frames"]) == 2

