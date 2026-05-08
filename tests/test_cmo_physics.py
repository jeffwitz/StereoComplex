from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stereocomplex.physics.cmo import (
    BrownConrady,
    CMOChannelRayField,
    CMOChannelSpec,
    CMOIntrinsics,
    CMOPlaneTargetSpec,
    CMOPolynomialChannelModel,
    CMOStereoSpec,
    PolynomialRayAberration,
    SensorWarp,
    Vignetting,
    cmo_polynomial_channel_parameters_from_spec,
    fit_cmo_stereo_model_and_poses_from_zernike_rayfields,
    generate_cmo_plane_dataset,
    pose_from_euler_xyz,
    project_cmo_points,
    project_cmo_target_corners,
    rays_from_cmo_pixels,
    render_cmo_channel_image,
)
import stereocomplex.physics.cmo as cmo_module
from stereocomplex.physics import (
    CentralBrownConradyModel,
    CentralPinholeModel,
    PhysicalModelSpec,
    PinholeParallelPlateModel,
    select_physical_model_from_rayfield,
)


def _small_cmo(*, vignetting: Vignetting | None = None) -> CMOStereoSpec:
    intr = CMOIntrinsics(width=96, height=72, fx=80.0, fy=80.0, cx=47.5, cy=35.5)
    vig = vignetting or Vignetting(strength=0.0)
    return CMOStereoSpec(
        left=CMOChannelSpec(
            name="left",
            intrinsics=intr,
            origin_world_mm=(-2.0, 0.0, 0.0),
            vignetting=vig,
        ),
        right=CMOChannelSpec(
            name="right",
            intrinsics=intr,
            origin_world_mm=(+2.0, 0.0, 0.0),
            vignetting=vig,
        ),
    )


def test_cmo_channel_rayfield_zero_aberration_center_rays() -> None:
    cmo = _small_cmo()

    O_l, d_l = CMOChannelRayField(cmo.left, cmo.common_aberration).ray(
        np.array([cmo.left.intrinsics.cx]),
        np.array([cmo.left.intrinsics.cy]),
    )
    O_r, d_r = CMOChannelRayField(cmo.right, cmo.common_aberration).ray(
        np.array([cmo.right.intrinsics.cx]),
        np.array([cmo.right.intrinsics.cy]),
    )

    assert np.allclose(O_l.reshape(-1, 3)[0], [-2.0, 0.0, 0.0])
    assert np.allclose(O_r.reshape(-1, 3)[0], [+2.0, 0.0, 0.0])
    assert np.allclose(d_l.reshape(-1, 3)[0], [0.0, 0.0, 1.0])
    assert np.allclose(d_r.reshape(-1, 3)[0], [0.0, 0.0, 1.0])


def test_cmo_dense_rays_use_same_channel_rayfield() -> None:
    cmo = _small_cmo()
    dense_O, dense_d = rays_from_cmo_pixels(cmo.left, cmo.common_aberration)
    u = np.array([0.0, 47.0, cmo.left.intrinsics.width - 1.0])
    v = np.array([0.0, 35.0, cmo.left.intrinsics.height - 1.0])
    sparse_O, sparse_d = CMOChannelRayField(cmo.left, cmo.common_aberration).ray(u, v)

    for idx, (uu, vv) in enumerate(zip(u.astype(int), v.astype(int), strict=True)):
        assert np.allclose(dense_O[vv, uu], sparse_O[idx])
        assert np.allclose(dense_d[vv, uu], sparse_d[idx])


def test_cmo_common_aberration_changes_both_channel_directions() -> None:
    intr = CMOIntrinsics(width=96, height=72, fx=80.0, fy=80.0, cx=47.5, cy=35.5)
    cmo = CMOStereoSpec(
        left=CMOChannelSpec("left", intr, (-2.0, 0.0, 0.0)),
        right=CMOChannelSpec("right", intr, (+2.0, 0.0, 0.0)),
        common_aberration=PolynomialRayAberration(coeff_x={"x2": 1.0e-2}),
    )
    u = np.array([80.0])
    v = np.array([40.0])

    _, d_left = CMOChannelRayField(cmo.left, cmo.common_aberration).ray(u, v)
    _, d_right = CMOChannelRayField(cmo.right, cmo.common_aberration).ray(u, v)
    _, d_ref = CMOChannelRayField(cmo.left, PolynomialRayAberration()).ray(u, v)

    assert not np.allclose(d_left, d_ref)
    assert np.allclose(d_left, d_right)


def test_cmo_dataset_writes_expected_files(tmp_path) -> None:
    cmo = _small_cmo()
    target = CMOPlaneTargetSpec(
        squares_x=5,
        squares_y=4,
        square_size_mm=2.0,
        pixels_per_square=12,
    )
    poses = [pose_from_euler_xyz(0.0, 0.0, 0.0, (0.0, 0.0, 45.0))]
    out_dir = tmp_path / "cmo"

    generate_cmo_plane_dataset(
        out_dir=out_dir,
        cmo=cmo,
        target=target,
        poses=poses,
        noise_std_gray=0.0,
        blur_sigma_px=0.0,
    )

    assert (out_dir / "meta.json").exists()
    assert (out_dir / "frames.jsonl").exists()
    assert (out_dir / "left" / "000000.png").exists()
    assert (out_dir / "right" / "000000.png").exists()
    assert (out_dir / "gt_charuco_corners.npz").exists()

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["generator"] == "stereocomplex.physics.cmo"
    gt = np.load(out_dir / "gt_charuco_corners.npz")
    assert gt["corner_id"].size > 0
    assert gt["uv_left_px"].shape[1] == 2
    assert gt["uv_right_px"].shape[1] == 2


def test_cmo_charuco_texture_does_not_silently_fallback_to_checker(monkeypatch) -> None:
    target = CMOPlaneTargetSpec(
        squares_x=5,
        squares_y=4,
        square_size_mm=2.0,
        pixels_per_square=12,
        pattern="charuco",
    )

    monkeypatch.setattr(cmo_module, "cv2", None)
    with pytest.raises(RuntimeError, match="ChArUco"):
        target.make_texture_u8()


def test_cmo_sparse_projection_matches_shared_pixel_to_ray_model() -> None:
    intr = CMOIntrinsics(width=120, height=90, fx=110.0, fy=112.0, cx=59.5, cy=44.5)
    channel = CMOChannelSpec(
        name="left",
        intrinsics=intr,
        origin_world_mm=(-2.5, 0.2, 0.0),
        distortion=BrownConrady(k1=-0.08, k2=0.02, p1=5.0e-4, p2=-3.0e-4),
        differential_aberration=PolynomialRayAberration(
            coeff_x={"x": 6.0e-4, "x2": 9.0e-4},
            coeff_y={"y": -5.0e-4, "xy": 7.0e-4},
        ),
        sensor_warp=SensorWarp(
            du_coeff_px={"xy": 0.18},
            dv_coeff_px={"x2": -0.11},
        ),
    )
    common = PolynomialRayAberration(coeff_x={"y2": -4.0e-4}, coeff_y={"xy": 3.0e-4})
    points = np.array(
        [
            [-3.0, -2.0, 60.0],
            [0.0, 0.0, 65.0],
            [4.0, 2.5, 70.0],
        ],
        dtype=np.float64,
    )

    uv = project_cmo_points(channel, common, points)
    origins, directions = CMOChannelRayField(channel, common).ray(uv[:, 0], uv[:, 1])
    distances = np.linalg.norm(np.cross(points - origins.reshape(-1, 3), directions.reshape(-1, 3)), axis=1)

    assert np.all(np.isfinite(uv))
    assert np.max(distances) < 1e-7


def test_vignetting_changes_image_intensity_but_not_projection() -> None:
    target = CMOPlaneTargetSpec(
        squares_x=5,
        squares_y=4,
        square_size_mm=2.0,
        pixels_per_square=12,
    )
    pose = pose_from_euler_xyz(0.0, 0.0, 0.0, (0.0, 0.0, 45.0))
    cmo_clear = _small_cmo(vignetting=Vignetting(strength=0.0))
    cmo_vignetted = _small_cmo(vignetting=Vignetting(strength=0.5, floor=0.2))

    gt_clear = project_cmo_target_corners(cmo_clear, target, pose)
    gt_vignetted = project_cmo_target_corners(cmo_vignetted, target, pose)

    assert np.array_equal(gt_clear["corner_id"], gt_vignetted["corner_id"])
    assert np.allclose(gt_clear["uv_left_px"], gt_vignetted["uv_left_px"])
    assert np.allclose(gt_clear["uv_right_px"], gt_vignetted["uv_right_px"])

    texture = target.make_texture_u8()
    img_clear = render_cmo_channel_image(
        cmo_clear,
        cmo_clear.left,
        target,
        pose,
        texture,
        background_gray=160,
    )
    img_vignetted = render_cmo_channel_image(
        cmo_vignetted,
        cmo_vignetted.left,
        target,
        pose,
        texture,
        background_gray=160,
    )

    assert img_vignetted.mean() < img_clear.mean()


def test_brown_physics_is_shared_by_cmo_projection() -> None:
    cmo = _small_cmo()
    distorted_left = replace(
        cmo.left,
        distortion=BrownConrady(k1=-0.1, k2=0.02, p1=1.0e-3, p2=-1.0e-3),
    )
    cmo_distorted = replace(cmo, left=distorted_left)
    target = CMOPlaneTargetSpec(
        squares_x=5,
        squares_y=4,
        square_size_mm=2.0,
        pixels_per_square=12,
    )
    pose = pose_from_euler_xyz(0.0, 0.0, 0.0, (0.0, 0.0, 45.0))

    gt_clear = project_cmo_target_corners(cmo, target, pose)
    gt_distorted = project_cmo_target_corners(cmo_distorted, target, pose)

    assert not np.allclose(gt_clear["uv_left_px"], gt_distorted["uv_left_px"])
    assert np.allclose(gt_clear["uv_right_px"], gt_distorted["uv_right_px"])


def test_cmo_polynomial_channel_model_recovers_cmo_rayfield() -> None:
    intr = CMOIntrinsics(width=160, height=120, fx=180.0, fy=180.0, cx=79.5, cy=59.5)
    channel = CMOChannelSpec(
        name="left",
        intrinsics=intr,
        origin_world_mm=(-3.0, 0.4, 0.0),
        distortion=BrownConrady(k1=-0.04, k2=0.01, p1=2.0e-4, p2=-3.0e-4),
        differential_aberration=PolynomialRayAberration(
            coeff_x={"x": 8.0e-4, "x2": 1.5e-3},
            coeff_y={"y": -6.0e-4, "xy": 1.0e-3},
        ),
    )
    target = CMOChannelRayField(channel)
    terms = CMOPolynomialChannelModel.default_terms()
    initial = np.zeros(8 + 2 * len(terms), dtype=np.float64)
    bounds = (
        np.r_[[-8.0, -8.0, -8.0, -0.5, -0.5, -0.05, -0.05, -0.5], -0.05 * np.ones(2 * len(terms))],
        np.r_[[+8.0, +8.0, +8.0, +0.5, +0.5, +0.05, +0.05, +0.5], +0.05 * np.ones(2 * len(terms))],
    )
    report = select_physical_model_from_rayfield(
        target_field=target,
        candidate_specs=[
            PhysicalModelSpec(
                name="cmo_polynomial_channel",
                model_class=CMOPolynomialChannelModel,
                initial_parameters=initial,
                bounds=bounds,
                model_kwargs={"cmo_image_size": (intr.width, intr.height), "aberration_terms": terms},
            )
        ],
        K=intr.as_K(),
        image_size=(intr.width, intr.height),
        grid_shape=(9, 7),
        full_grid_weight=0.0,
    )

    result = report.candidates[0]
    assert result.rms_mm < 1e-6
    assert abs(result.parameter_dict["origin_x_mm"] + 3.0) < 1e-5
    assert abs(result.parameter_dict["origin_y_mm"] - 0.4) < 1e-5
    # origin_z was 0.0 in the oracle, should be recovered to near-zero.
    assert abs(result.parameter_dict["origin_z_mm"] - 0.0) < 1e-5


def test_model_selection_prefers_cmo_on_cmo_oracle() -> None:
    intr = CMOIntrinsics(width=160, height=120, fx=180.0, fy=180.0, cx=79.5, cy=59.5)
    channel = CMOChannelSpec(
        name="left",
        intrinsics=intr,
        origin_world_mm=(-3.0, 0.0, 0.0),
        distortion=BrownConrady(k1=-0.04, k2=0.01, p1=2.0e-4, p2=-3.0e-4),
        differential_aberration=PolynomialRayAberration(
            coeff_x={"x": 8.0e-4, "x2": 1.5e-3},
            coeff_y={"y": -6.0e-4, "xy": 1.0e-3},
        ),
    )
    target = CMOChannelRayField(channel)
    terms = CMOPolynomialChannelModel.default_terms()
    cmo_initial = np.zeros(8 + 2 * len(terms), dtype=np.float64)
    cmo_bounds = (
        np.r_[[-8.0, -8.0, -50.0, -0.5, -0.5, -0.05, -0.05, -0.5], -0.05 * np.ones(2 * len(terms))],
        np.r_[[+8.0, +8.0, +50.0, +0.5, +0.5, +0.05, +0.05, +0.5], +0.05 * np.ones(2 * len(terms))],
    )
    report = select_physical_model_from_rayfield(
        target_field=target,
        candidate_specs=[
            PhysicalModelSpec("central_pinhole", CentralPinholeModel, np.zeros(0)),
            PhysicalModelSpec(
                "central_brown_conrady",
                CentralBrownConradyModel,
                np.zeros(5),
                bounds=(
                    np.array([-1.0, -1.0, -0.1, -0.1, -1.0]),
                    np.array([1.0, 1.0, 0.1, 0.1, 1.0]),
                ),
            ),
            PhysicalModelSpec(
                "pinhole_parallel_plate",
                PinholeParallelPlateModel,
                np.array([0.0, 0.0, 8.0]),
                bounds=(np.array([-30.0, -30.0, 0.0]), np.array([30.0, 30.0, 50.0])),
                model_kwargs={"eta": 1.5, "d1_mm": 80.0},
            ),
            PhysicalModelSpec(
                "cmo_polynomial_channel",
                CMOPolynomialChannelModel,
                cmo_initial,
                bounds=cmo_bounds,
                model_kwargs={"cmo_image_size": (intr.width, intr.height), "aberration_terms": terms},
            ),
        ],
        K=intr.as_K(),
        image_size=(intr.width, intr.height),
        grid_shape=(9, 7),
        full_grid_weight=0.0,
    )

    by_name = {candidate.model_name: candidate for candidate in report.candidates}
    assert report.best_by_bic == "cmo_polynomial_channel"
    assert by_name["cmo_polynomial_channel"].rms_mm < by_name["central_brown_conrady"].rms_mm
    assert by_name["cmo_polynomial_channel"].rms_mm < by_name["pinhole_parallel_plate"].rms_mm


def test_cmo_stereo_ba_recovers_effective_channels_and_poses_from_rayfields() -> None:
    intr = CMOIntrinsics(width=160, height=120, fx=180.0, fy=180.0, cx=79.5, cy=59.5)
    common = PolynomialRayAberration(coeff_x={"x2": 7.0e-4}, coeff_y={"xy": 4.0e-4})
    cmo = CMOStereoSpec(
        left=CMOChannelSpec(
            name="left",
            intrinsics=intr,
            origin_world_mm=(-3.0, 0.2, 0.0),
            distortion=BrownConrady(k1=-0.04, k2=0.01, p1=2.0e-4, p2=-3.0e-4),
            differential_aberration=PolynomialRayAberration(
                coeff_x={"x": 5.0e-4},
                coeff_y={"y": -3.0e-4},
            ),
        ),
        right=CMOChannelSpec(
            name="right",
            intrinsics=intr,
            origin_world_mm=(3.0, -0.1, 0.0),
            distortion=BrownConrady(k1=-0.035, k2=0.008, p1=-2.0e-4, p2=2.0e-4),
            differential_aberration=PolynomialRayAberration(
                coeff_x={"x": -4.0e-4},
                coeff_y={"y": 3.0e-4},
            ),
        ),
        common_aberration=common,
    )
    target = CMOPlaneTargetSpec(
        squares_x=5,
        squares_y=4,
        square_size_mm=2.0,
        pixels_per_square=12,
        pattern="charuco",
    )
    poses = [
        pose_from_euler_xyz(0.0, 0.0, 0.0, (0.0, 0.0, 55.0)),
        pose_from_euler_xyz(0.04, -0.03, 0.02, (1.2, -0.8, 60.0)),
    ]
    ids, xy_all = target.inner_corners_local_mm()
    object_frames = []
    left_frames = []
    right_frames = []
    for pose in poses:
        gt = project_cmo_target_corners(cmo, target, pose)
        object_frames.append(xy_all[gt["corner_id"]])
        left_frames.append(gt["uv_left_px"])
        right_frames.append(gt["uv_right_px"])

    terms = CMOPolynomialChannelModel.default_terms()
    left_truth = cmo_polynomial_channel_parameters_from_spec(cmo.left, common, terms)
    right_truth = cmo_polynomial_channel_parameters_from_spec(cmo.right, common, terms)
    left_initial = left_truth.copy()
    right_initial = right_truth.copy()
    left_initial[:3] += np.array([0.4, -0.25, 0.0])
    right_initial[:3] += np.array([-0.35, 0.2, 0.0])
    left_initial[3:8] = 0.0
    right_initial[3:8] = 0.0
    pose_initials = [
        pose_from_euler_xyz(0.01, -0.01, 0.005, (0.15, -0.12, 55.3)),
        pose_from_euler_xyz(0.05, -0.04, 0.025, (1.35, -0.95, 60.2)),
    ]

    result = fit_cmo_stereo_model_and_poses_from_zernike_rayfields(
        left_field=CMOChannelRayField(cmo.left, cmo.common_aberration),
        right_field=CMOChannelRayField(cmo.right, cmo.common_aberration),
        K=intr.as_K(),
        image_size=(intr.width, intr.height),
        object_points=object_frames,
        left_pixels=left_frames,
        right_pixels=right_frames,
        pose_initials=pose_initials,
        initial_left_parameters=left_initial,
        initial_right_parameters=right_initial,
        aberration_terms=terms,
        rayfield_weight=0.5,
        pose_regularization=0.0,
        max_nfev=300,
    )

    assert result.success
    assert result.incidence_rms_mm < 1e-5
    assert result.left_rayfield_rms_mm < 1e-4
    assert result.right_rayfield_rms_mm < 1e-4
    assert abs(result.left_model.origin_x_mm - cmo.left.origin[0]) < 1e-4
    assert abs(result.right_model.origin_x_mm - cmo.right.origin[0]) < 1e-4
    assert abs(result.left_model.origin_z_mm) < 1e-4
    assert abs(result.right_model.origin_z_mm) < 1e-4
    for fitted, truth in zip(result.poses, poses, strict=True):
        assert np.linalg.norm(fitted.t - truth.t) < 1e-4
        d_rot = Rotation.from_matrix(fitted.R @ truth.R.T).magnitude()
        assert np.degrees(d_rot) < 1e-4
