from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stereocomplex.calibration.fit_zernike_origin_field import (
    StereoZernikeOriginFieldFitResult,
    fit_stereo_zernike_origin_field,
)
from stereocomplex.api.corner_refinement import RefineMethod
from stereocomplex.metrics.rayfield_metrics import (
    RayfieldComparisonReport,
    compare_rayfields_on_planes,
)
from stereocomplex.metrics.reconstruction_metrics import (
    OracleReconstructionFloorReport,
    ReconstructionComparisonReport,
    ReconstructionErrorReport,
    compare_3d_reconstruction_with_without_origin_field,
    oracle_reconstruction_floor_report,
)
from stereocomplex.rayfields.zernike_origin_field import ZernikeOriginFieldConfig
from stereocomplex.synthetic.parallel_plate import (
    ParallelPlateSyntheticParams,
    SyntheticStereoDataset,
    generate_parallel_plate_stereo_dataset,
)
from stereocomplex.synthetic.parallel_plate_images import (
    ParallelPlateImageRenderParams,
    RenderedParallelPlateImageDataset,
    charuco_inner_corners_object_points,
    detected_observations_from_rendered_parallel_plate,
    render_parallel_plate_charuco_images,
)
from stereocomplex.api.calibration import CharucoBoardSpec


@dataclass(frozen=True)
class BenchmarkReport:
    fit_result: StereoZernikeOriginFieldFitResult
    reconstruction_comparison: ReconstructionComparisonReport
    oracle_floor: OracleReconstructionFloorReport
    left_rayfield_comparison: RayfieldComparisonReport
    right_rayfield_comparison: RayfieldComparisonReport


@dataclass(frozen=True)
class RenderedImageBenchmarkReport:
    fit_result: StereoZernikeOriginFieldFitResult
    reconstruction_comparison: ReconstructionComparisonReport
    oracle_detected: ReconstructionErrorReport
    rendered: RenderedParallelPlateImageDataset
    detected_dataset: SyntheticStereoDataset
    method2d: RefineMethod
    n_common_corners: int
    n_frames: int
    n_points_total: int


def make_grid_board(nx: int = 7, ny: int = 5, spacing: float = 24.0) -> np.ndarray:
    xs = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) * float(spacing)
    ys = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * float(spacing)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.reshape(-1), yy.reshape(-1), np.zeros(nx * ny)])


def make_transform(R: np.ndarray | None = None, t: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(float(deg))
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]], dtype=np.float64)


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(float(deg))
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=np.float64)


def make_default_parallel_plate_dataset(noise_std_px: float = 0.0) -> SyntheticStereoDataset:
    image_size = (640, 480)
    K_left = np.array([[620.0, 0.0, 319.5], [0.0, 620.0, 239.5], [0.0, 0.0, 1.0]])
    K_right = K_left.copy()
    baseline = 90.0
    T_left_world = np.eye(4, dtype=np.float64)
    T_right_world = make_transform(t=np.array([-baseline, 0.0, 0.0]))
    board_points = make_grid_board()
    board_poses = [
        make_transform(t=np.array([-28.0, -18.0, 650.0])),
        make_transform(t=np.array([24.0, 16.0, 720.0])),
        make_transform(t=np.array([0.0, -24.0, 800.0])),
        make_transform(t=np.array([34.0, 22.0, 900.0])),
    ]
    plate_left = ParallelPlateSyntheticParams(
        eta=1.5, thickness=16.0, alpha_deg=13.0, beta_deg=5.0, d1=70.0
    )
    plate_right = ParallelPlateSyntheticParams(
        eta=1.5, thickness=14.0, alpha_deg=10.0, beta_deg=7.0, d1=75.0
    )
    return generate_parallel_plate_stereo_dataset(
        object_points=board_points,
        board_poses=board_poses,
        K_left=K_left,
        K_right=K_right,
        T_left_world=T_left_world,
        T_right_world=T_right_world,
        plate_left=plate_left,
        plate_right=plate_right,
        image_size=image_size,
        noise_std_px=noise_std_px,
        keep_oracle_rayfields=True,
    )


def _board_visibility_mask(
    board_points: np.ndarray,
    tx: float,
    ty: float,
    tz: float,
    baseline: float,
    K: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Return boolean mask of board points projecting inside both cameras (pinhole approx)."""
    W, H = image_size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = board_points[:, 0] + tx
    y = board_points[:, 1] + ty
    u_left = fx * x / tz + cx
    v = fy * y / tz + cy
    u_right = fx * (x - baseline) / tz + cx
    in_left = (u_left >= 0.0) & (u_left <= W - 1) & (v >= 0.0) & (v <= H - 1)
    in_right = (u_right >= 0.0) & (u_right <= W - 1) & (v >= 0.0) & (v <= H - 1)
    return in_left & in_right


def make_parallel_plate_wide_coverage_dataset(noise_std_px: float = 0.0) -> SyntheticStereoDataset:
    """
    Dataset with a larger board and wider pose coverage for rayfield support plots.

    The default dataset is intentionally small and fast because it is used by
    multiple regression tests.  This variant is used for physical rayfield
    interpretation figures where the observed pixel support should cover much
    more of the image.

    Six central poses keep the full 11x9 board within both cameras.  Four
    axis-aligned paired edge poses cover the top/bottom/right image borders.
    One additional left-only monocular frame (empty right_pixels sentinel)
    fills the left-edge gap that paired stereo geometry cannot reach:
    u_min_paired = baseline * fx / z = 90 * 620 / 650 ≈ 86 px.  The total is
    11 frames; only the monocular frame contributes no right-camera residuals.
    """
    image_size = (640, 480)
    K_left = np.array([[620.0, 0.0, 319.5], [0.0, 620.0, 239.5], [0.0, 0.0, 1.0]])
    K_right = K_left.copy()
    baseline = 90.0
    T_left_world = np.eye(4, dtype=np.float64)
    T_right_world = make_transform(t=np.array([-baseline, 0.0, 0.0]))
    all_board_points = make_grid_board(nx=11, ny=9, spacing=38.0)
    plate_left = ParallelPlateSyntheticParams(
        eta=1.5, thickness=16.0, alpha_deg=13.0, beta_deg=5.0, d1=70.0
    )
    plate_right = ParallelPlateSyntheticParams(
        eta=1.5, thickness=14.0, alpha_deg=10.0, beta_deg=7.0, d1=75.0
    )

    # Central poses — full board visible in both cameras
    central_poses = [
        make_transform(R=_rot_y(-8.0) @ _rot_x(4.0), t=np.array([-62.0, -38.0, 720.0])),
        make_transform(R=_rot_y(8.0) @ _rot_x(-5.0), t=np.array([72.0, 42.0, 740.0])),
        make_transform(R=_rot_y(0.0) @ _rot_x(8.0), t=np.array([0.0, -62.0, 760.0])),
        make_transform(R=_rot_y(-7.0) @ _rot_x(-4.0), t=np.array([84.0, -6.0, 800.0])),
        make_transform(R=_rot_y(9.0) @ _rot_x(4.0), t=np.array([-78.0, 58.0, 820.0])),
        make_transform(R=_rot_y(-4.0) @ _rot_x(10.0), t=np.array([16.0, 58.0, 700.0])),
    ]
    central_dataset = generate_parallel_plate_stereo_dataset(
        object_points=all_board_points,
        board_poses=central_poses,
        K_left=K_left,
        K_right=K_right,
        T_left_world=T_left_world,
        T_right_world=T_right_world,
        plate_left=plate_left,
        plate_right=plate_right,
        image_size=image_size,
        noise_std_px=noise_std_px,
        keep_oracle_rayfields=True,
    )

    # Paired edge poses.  The mask uses the translation only (pinhole approx).
    # Rotations are chosen so that after the plate model the extreme board
    # points remain inside the image (plate shifts can push marginal pinhole
    # pixels outside the sensor boundary and trigger boundary-clipping in
    # project_point_with_parallel_plate, corrupting the observation).
    # Left  (tx=-145, rot_y=+6°): extreme left board column shifts from
    #   u_pinhole≈0 to u_pinhole≈10 px, u_plate stays above 0.
    # Right (tx=+143, no rotation): board_x=+190 → u_plate≈638.6 px (safe).
    # Top   (ty=-99, rot_x=-6°): extreme top row shifts from v_pinhole≈0.4 px
    #   to v_plate≈6.6 px (safe without clipping).
    # Bottom (ty=+99, rot_x=+6°): symmetric.
    edge_specs = [
        (make_transform(R=_rot_y(6.0),  t=np.array([-145.0,   0.0, 650.0])), -145.0,   0.0, 650.0),
        (make_transform(t=np.array([ 143.0,   0.0, 650.0])),                   143.0,   0.0, 650.0),
        (make_transform(R=_rot_x(-6.0), t=np.array([   0.0, -99.0, 650.0])),    0.0, -99.0, 650.0),
        (make_transform(R=_rot_x(6.0),  t=np.array([   0.0,  99.0, 650.0])),    0.0,  99.0, 650.0),
    ]

    board_poses = list(central_dataset.board_poses)
    left_pixels: list[np.ndarray] = list(central_dataset.left_pixels)
    right_pixels: list[np.ndarray] = list(central_dataset.right_pixels)
    per_frame_obj: list[np.ndarray] = [all_board_points] * len(central_poses)

    for pose, tx, ty, tz in edge_specs:
        mask = _board_visibility_mask(all_board_points, tx, ty, tz, baseline, K_left, image_size)
        visible_pts = all_board_points[mask]
        edge_ds = generate_parallel_plate_stereo_dataset(
            object_points=visible_pts,
            board_poses=[pose],
            K_left=K_left,
            K_right=K_right,
            T_left_world=T_left_world,
            T_right_world=T_right_world,
            plate_left=plate_left,
            plate_right=plate_right,
            image_size=image_size,
            noise_std_px=noise_std_px,
            keep_oracle_rayfields=True,
        )
        board_poses.append(pose)
        left_pixels.append(edge_ds.left_pixels[0])
        right_pixels.append(edge_ds.right_pixels[0])
        per_frame_obj.append(visible_pts)

    # Left-only monocular frame: board at tx=-138 mm with ALL columns.
    # tx=-138 (not -145) keeps board_x=-190 → u_plate≈1.8 px (safe); at -145
    # the plate model clips that column to u=0, corrupting the observation.
    # The left camera sees u_left ∈ [~2, ~370] px — fills the [0, ~86] gap
    # that paired stereo cannot reach (u_min_paired ≈ 86 px at z=650 mm).
    # right_pixels is an empty (0×2) sentinel; fit_stereo_zernike_origin_field
    # skips right-camera residuals for frames where right_pixels.shape[0] == 0.
    left_only_pose = make_transform(t=np.array([-138.0, 0.0, 650.0]))
    left_only_ds = generate_parallel_plate_stereo_dataset(
        object_points=all_board_points,
        board_poses=[left_only_pose],
        K_left=K_left,
        K_right=K_right,
        T_left_world=T_left_world,
        T_right_world=T_right_world,
        plate_left=plate_left,
        plate_right=plate_right,
        image_size=image_size,
        noise_std_px=noise_std_px,
        keep_oracle_rayfields=False,
    )
    board_poses.append(left_only_pose)
    left_pixels.append(left_only_ds.left_pixels[0])
    right_pixels.append(np.zeros((0, 2), dtype=np.float64))
    per_frame_obj.append(all_board_points)

    return SyntheticStereoDataset(
        object_points=all_board_points,
        board_poses=board_poses,
        left_pixels=left_pixels,
        right_pixels=right_pixels,
        per_frame_object_points=per_frame_obj,
        K_left=K_left,
        K_right=K_right,
        T_left_world=T_left_world,
        T_right_world=T_right_world,
        image_size=image_size,
        oracle_left_params=plate_left,
        oracle_right_params=plate_right,
    )


# Frame indices for the extended 10-frame dataset
EXTENDED_TRAIN_FRAMES: list[int] = list(range(8))
EXTENDED_HOLDOUT_FRAMES: list[int] = [8, 9]


def make_parallel_plate_extended_dataset(noise_std_px: float = 0.0) -> SyntheticStereoDataset:
    """
    10-frame dataset for hold-out generalization validation.

    Frames 0–7 (``EXTENDED_TRAIN_FRAMES``) are used for fitting.
    Frames 8–9 (``EXTENDED_HOLDOUT_FRAMES``) are reserved for held-out evaluation
    and must never enter the BA.  Same oracle as `make_default_parallel_plate_dataset`.
    """
    image_size = (640, 480)
    K_left = np.array([[620.0, 0.0, 319.5], [0.0, 620.0, 239.5], [0.0, 0.0, 1.0]])
    K_right = K_left.copy()
    baseline = 90.0
    T_left_world = np.eye(4, dtype=np.float64)
    T_right_world = make_transform(t=np.array([-baseline, 0.0, 0.0]))
    board_points = make_grid_board()
    board_poses = [
        # Training frames (0-7): varied XY offset and depth
        make_transform(t=np.array([-28.0, -18.0, 650.0])),
        make_transform(t=np.array([24.0,  16.0,  720.0])),
        make_transform(t=np.array([0.0,  -24.0,  800.0])),
        make_transform(t=np.array([34.0,  22.0,  900.0])),
        make_transform(t=np.array([-35.0, 25.0,  680.0])),
        make_transform(t=np.array([15.0, -28.0,  760.0])),
        make_transform(t=np.array([-12.0, 10.0,  840.0])),
        make_transform(t=np.array([28.0, -12.0,  780.0])),
        # Hold-out frames (8-9): unseen positions not in training set
        make_transform(t=np.array([-42.0, -8.0,  710.0])),
        make_transform(t=np.array([38.0,  28.0,  860.0])),
    ]
    plate_left = ParallelPlateSyntheticParams(
        eta=1.5, thickness=16.0, alpha_deg=13.0, beta_deg=5.0, d1=70.0
    )
    plate_right = ParallelPlateSyntheticParams(
        eta=1.5, thickness=14.0, alpha_deg=10.0, beta_deg=7.0, d1=75.0
    )
    return generate_parallel_plate_stereo_dataset(
        object_points=board_points,
        board_poses=board_poses,
        K_left=K_left,
        K_right=K_right,
        T_left_world=T_left_world,
        T_right_world=T_right_world,
        plate_left=plate_left,
        plate_right=plate_right,
        image_size=image_size,
        noise_std_px=noise_std_px,
        keep_oracle_rayfields=True,
    )


def make_default_parallel_plate_charuco_board() -> CharucoBoardSpec:
    """Return the ChArUco board used by the rendered non-central image demo."""
    return CharucoBoardSpec(
        squares_x=12,
        squares_y=9,
        square_size_mm=28.0,
        marker_size_mm=19.6,
        aruco_dictionary="DICT_4X4_1000",
        adaptive_thresh_win_size_max=300,
        corner_refinement_win_size=5,
        corner_refinement_max_iterations=50,
        corner_refinement_min_accuracy=1e-3,
    )


def make_default_parallel_plate_charuco_dataset(
    noise_std_px: float = 0.0,
) -> SyntheticStereoDataset:
    """
    Default parallel-plate dataset whose object points are ChArUco inner corners.

    This is the geometric counterpart of the rendered-image benchmark. The same
    board poses and optical oracle are used, but observations are exact projected
    ChArUco inner corners rather than OpenCV detections.
    """
    board = make_default_parallel_plate_charuco_board()
    _ids, board_points = charuco_inner_corners_object_points(board)
    image_size = (640, 480)
    K_left = np.array([[620.0, 0.0, 319.5], [0.0, 620.0, 239.5], [0.0, 0.0, 1.0]])
    K_right = K_left.copy()
    baseline = 90.0
    T_left_world = np.eye(4, dtype=np.float64)
    T_right_world = make_transform(t=np.array([-baseline, 0.0, 0.0]))
    board_poses = [
        # Central poses — board fully inside image
        make_transform(R=_rot_y(-10.0) @ _rot_x(5.0), t=np.array([-34.0, -24.0, 650.0])),
        make_transform(R=_rot_y(8.0) @ _rot_x(-6.0), t=np.array([28.0, 20.0, 690.0])),
        make_transform(R=_rot_y(0.0) @ _rot_x(8.0), t=np.array([0.0, -28.0, 730.0])),
        make_transform(R=_rot_y(-7.0) @ _rot_x(-4.0), t=np.array([36.0, -8.0, 760.0])),
        make_transform(R=_rot_y(11.0) @ _rot_x(4.0), t=np.array([-24.0, 26.0, 800.0])),
        make_transform(R=_rot_y(-4.0) @ _rot_x(10.0), t=np.array([14.0, 12.0, 660.0])),
        make_transform(R=_rot_y(6.0) @ _rot_x(-10.0), t=np.array([-18.0, -14.0, 710.0])),
        make_transform(R=_rot_y(-12.0) @ _rot_x(0.0), t=np.array([20.0, -24.0, 780.0])),
        make_transform(R=_rot_y(4.0) @ _rot_x(12.0), t=np.array([-36.0, 12.0, 740.0])),
        make_transform(R=_rot_y(0.0) @ _rot_x(-8.0), t=np.array([6.0, 28.0, 820.0])),
        # Edge-coverage poses — board partially outside image so that the union of
        # observations spans the full 640×480 pixel area, eliminating extrapolation
        # at image edges for the Zernike fit.  Corner poses are intentionally
        # omitted: combining extreme X and Y offsets simultaneously pushes too many
        # ArUco markers outside the image, preventing ChArUco corner identification.
        make_transform(R=_rot_y(6.0), t=np.array([-185.0, 0.0, 650.0])),   # left edge
        make_transform(R=_rot_y(-6.0), t=np.array([185.0, 0.0, 650.0])),   # right edge
        make_transform(R=_rot_x(-6.0), t=np.array([0.0, -125.0, 650.0])),  # top edge
        make_transform(R=_rot_x(6.0), t=np.array([0.0, 125.0, 650.0])),    # bottom edge
    ]
    plate_left = ParallelPlateSyntheticParams(
        eta=1.5, thickness=16.0, alpha_deg=13.0, beta_deg=5.0, d1=70.0
    )
    plate_right = ParallelPlateSyntheticParams(
        eta=1.5, thickness=14.0, alpha_deg=10.0, beta_deg=7.0, d1=75.0
    )
    return generate_parallel_plate_stereo_dataset(
        object_points=board_points,
        board_poses=board_poses,
        K_left=K_left,
        K_right=K_right,
        T_left_world=T_left_world,
        T_right_world=T_right_world,
        plate_left=plate_left,
        plate_right=plate_right,
        image_size=image_size,
        noise_std_px=noise_std_px,
        keep_oracle_rayfields=True,
    )


def run_parallel_plate_origin_field_benchmark(
    max_order: int = 4,
    noise_std_px: float = 0.05,
    optimize_directions: bool = False,
    optimize_board_poses: bool = False,
    optimize_stereo_extrinsics: bool = False,
    pose_regularization: float = 0.0,
    rig_regularization: float = 0.0,
) -> BenchmarkReport:
    dataset = make_default_parallel_plate_dataset(noise_std_px=noise_std_px)
    dataset_clean = make_default_parallel_plate_dataset(noise_std_px=0.0)
    config_left = ZernikeOriginFieldConfig(image_size=dataset.image_size, max_order=max_order)
    config_right = ZernikeOriginFieldConfig(image_size=dataset.image_size, max_order=max_order)
    fit_result = fit_stereo_zernike_origin_field(
        observations=dataset,
        K_left=dataset.K_left,
        K_right=dataset.K_right,
        T_right_left_initial=dataset.T_right_left,
        board_poses_initial=dataset.board_poses,
        config_left=config_left,
        config_right=config_right,
        optimize_board_poses=optimize_board_poses,
        optimize_stereo_extrinsics=optimize_stereo_extrinsics,
        optimize_directions=optimize_directions,
        robust_loss="huber",
        regularization=1e-3,
        direction_regularization=1e-2,
        pose_regularization=pose_regularization,
        rig_regularization=rig_regularization,
    )
    reconstruction_comparison = compare_3d_reconstruction_with_without_origin_field(
        dataset=dataset,
        central_model_result=None,
        origin_field_result=fit_result,
    )
    oracle_floor = oracle_reconstruction_floor_report(dataset, dataset_clean=dataset_clean)
    if dataset.oracle_left_ray_function is None or dataset.oracle_right_ray_function is None:
        raise RuntimeError("benchmark dataset was generated without oracle rayfields")
    left_rayfield_comparison = compare_rayfields_on_planes(
        fitted_field=fit_result.left_field,
        oracle_ray_function=dataset.oracle_left_ray_function,
        image_size=dataset.image_size,
        z_planes=(100.0, 1000.0),
        grid_shape=(25, 19),
    )
    right_rayfield_comparison = compare_rayfields_on_planes(
        fitted_field=fit_result.right_field,
        oracle_ray_function=dataset.oracle_right_ray_function,
        image_size=dataset.image_size,
        z_planes=(100.0, 1000.0),
        grid_shape=(25, 19),
    )
    return BenchmarkReport(
        fit_result=fit_result,
        reconstruction_comparison=reconstruction_comparison,
        oracle_floor=oracle_floor,
        left_rayfield_comparison=left_rayfield_comparison,
        right_rayfield_comparison=right_rayfield_comparison,
    )


def run_parallel_plate_rendered_image_benchmark(
    out_dir,
    max_order: int = 3,
    render_params: ParallelPlateImageRenderParams | None = None,
    method2d: RefineMethod = "raw",
) -> RenderedImageBenchmarkReport:
    """
    Render non-central ChArUco images, detect them with OpenCV, then run full BA.

    This is the first end-to-end image front-end check for the parallel-plate
    oracle. The fit does not receive oracle plate parameters; it only receives
    detected 2D ChArUco corners, the board model, and a central initialization.
    """
    dataset = make_default_parallel_plate_charuco_dataset(noise_std_px=0.0)
    board = make_default_parallel_plate_charuco_board()
    rendered = render_parallel_plate_charuco_images(dataset, board, out_dir, params=render_params)
    detected = detected_observations_from_rendered_parallel_plate(
        rendered,
        min_common_corners=12,
        method2d=method2d,
    )
    config = ZernikeOriginFieldConfig(image_size=detected.image_size, max_order=max_order)
    fit_result = fit_stereo_zernike_origin_field(
        observations=detected,
        K_left=detected.K_left,
        K_right=detected.K_right,
        T_right_left_initial=detected.T_right_left,
        board_poses_initial=detected.board_poses,
        config_left=config,
        config_right=config,
        optimize_board_poses=True,
        optimize_stereo_extrinsics=True,
        optimize_directions=True,
        robust_loss="huber",
        regularization=1e-5,
        direction_regularization=1e-2,
        pose_regularization=100.0,
        rig_regularization=1000.0,
        max_nfev=200,
    )
    reconstruction_comparison = compare_3d_reconstruction_with_without_origin_field(
        dataset=detected,
        central_model_result=None,
        origin_field_result=fit_result,
    )
    oracle_detected = oracle_reconstruction_floor_report(detected).oracle_observed_pixels
    n_frames = len(detected.left_pixels)
    # n_common_corners = min per-frame detected corner count (each frame may see a
    # different subset when edge-coverage poses partially extend outside the image)
    n_common = min(pix.shape[0] for pix in detected.left_pixels)
    n_total = sum(pix.shape[0] for pix in detected.left_pixels)
    return RenderedImageBenchmarkReport(
        fit_result=fit_result,
        reconstruction_comparison=reconstruction_comparison,
        oracle_detected=oracle_detected,
        rendered=rendered,
        detected_dataset=detected,
        method2d=method2d,
        n_common_corners=n_common,
        n_frames=n_frames,
        n_points_total=n_total,
    )
