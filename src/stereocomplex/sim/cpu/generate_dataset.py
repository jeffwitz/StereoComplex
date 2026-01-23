from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from stereocomplex.core.distortion import BrownDistortion, brown_to_dict
from stereocomplex.core.geometry import PinholeCamera, pixel_grid_um, sensor_um_to_pixel
from stereocomplex.meta import parse_view_meta
from stereocomplex.sim.cpu.effects import fwhm_to_sigma, gaussian_blur_edge_varying_u8, gaussian_blur_u8
from stereocomplex.sim.patterns.charuco import CharucoSpec, generate_charuco_texture


def generate_cpu_dataset(
    out_root: Path,
    scenes: int,
    frames_per_scene: int,
    width: int,
    height: int,
    pattern: str = "auto",
    tex_interp: str = "linear",
    distort: str = "none",
    distort_strength: float = 0.0,
    image_format: str = "png",
    outside_mask: str = "none",
    blur_fwhm_um: float = 0.0,
    blur_fwhm_px: float = 0.0,
    blur_edge_factor: float = 1.0,
    blur_edge_start: float = 0.6,
    blur_edge_power: float = 2.0,
    noise_std: float = 0.02,
    seed: int = 0,
    pitch_um_override: float | None = None,
    f_um_override: float | None = None,
    tz_nominal_mm_override: float | None = None,
    tz_schedule_mm: list[float] | None = None,
    baseline_mm_override: float | None = None,
    board_squares_x_override: int | None = None,
    board_squares_y_override: int | None = None,
    board_square_size_mm_override: float | None = None,
    board_marker_size_mm_override: float | None = None,
    board_pixels_per_square_override: int | None = None,
    z_only_mode: bool = False,
) -> None:
    rng = np.random.default_rng(seed)
    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "train").mkdir(exist_ok=True)
    (out_root / "val").mkdir(exist_ok=True)
    (out_root / "test").mkdir(exist_ok=True)

    manifest = {"schema_version": "stereocomplex.dataset.v0", "generator": "cpu_mvp", "seed": seed}
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Simple split: all scenes in train for MVP; user can reshuffle later.
    for scene_idx in range(scenes):
        scene_dir = out_root / "train" / f"scene_{scene_idx:04d}"
        _generate_scene(
            scene_dir,
            frames_per_scene,
            width,
            height,
            pattern,
            tex_interp,
            distort,
            distort_strength,
            image_format,
            outside_mask,
            blur_fwhm_um,
            blur_fwhm_px,
            blur_edge_factor,
            blur_edge_start,
            blur_edge_power,
            noise_std,
            pitch_um_override,
            f_um_override,
            tz_nominal_mm_override,
            tz_schedule_mm,
            baseline_mm_override,
            board_squares_x_override,
            board_squares_y_override,
            board_square_size_mm_override,
            board_marker_size_mm_override,
            board_pixels_per_square_override,
            rng,
            z_only_mode,
        )


def _generate_scene(
    scene_dir: Path,
    frames_per_scene: int,
    width: int,
    height: int,
    pattern: str,
    tex_interp: str,
    distort: str,
    distort_strength: float,
    image_format: str,
    outside_mask: str,
    blur_fwhm_um: float,
    blur_fwhm_px: float,
    blur_edge_factor: float,
    blur_edge_start: float,
    blur_edge_power: float,
    noise_std: float,
    pitch_um_override: float | None,
    f_um_override: float | None,
    tz_nominal_mm_override: float | None,
    tz_schedule_mm: list[float] | None,
    baseline_mm_override: float | None,
    board_squares_x_override: int | None,
    board_squares_y_override: int | None,
    board_square_size_mm_override: float | None,
    board_marker_size_mm_override: float | None,
    board_pixels_per_square_override: int | None,
    rng: np.random.Generator,
    z_only_mode: bool,
) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    left_dir = scene_dir / "left"
    right_dir = scene_dir / "right"
    left_dir.mkdir(exist_ok=True)
    right_dir.mkdir(exist_ok=True)

    # Meta: keep it explicit and compatible OptiX later.
    pitch_um = float(pitch_um_override) if pitch_um_override is not None else float(rng.uniform(1.4, 6.5))
    view = {
        "schema_version": "stereocomplex.meta.v0",
        "sensor": {"pixel_pitch_um": pitch_um, "binning_xy": [1, 1]},
        "preprocess": {"crop_xywh_px": [0, 0, width, height], "resize_xy": [1.0, 1.0]},
        "image": {"width_px": width, "height_px": height, "bit_depth": 8, "gamma": 1.0},
    }

    stereo_meta = {"left": view, "right": view}
    # Camera model (pinhole in camera frame, sensor-plane in mm derived from meta).
    f_um = float(f_um_override) if f_um_override is not None else float(rng.uniform(4000.0, 40000.0))

    # Pick a nominal working distance, then size the board so it is actually visible.
    tz_nominal = (
        float(tz_nominal_mm_override) if tz_nominal_mm_override is not None else float(rng.uniform(200.0, 5000.0))
    )
    if tz_schedule_mm is not None:
        if len(tz_schedule_mm) != int(frames_per_scene):
            raise ValueError("tz_schedule_mm length must match frames_per_scene")
        tz_nominal = float(np.mean(np.asarray(tz_schedule_mm, dtype=np.float64)))
    f_mm = f_um / 1000.0
    sensor_half_w_mm = ((width - 1) / 2.0) * (pitch_um / 1000.0)
    sensor_half_h_mm = ((height - 1) / 2.0) * (pitch_um / 1000.0)
    visible_w_mm = 2.0 * tz_nominal * (sensor_half_w_mm / f_mm)
    visible_h_mm = 2.0 * tz_nominal * (sensor_half_h_mm / f_mm)

    if (
        board_squares_x_override is not None
        and board_squares_y_override is not None
        and board_square_size_mm_override is not None
    ):
        cols = int(board_squares_x_override)
        rows = int(board_squares_y_override)
        square_size_mm = float(board_square_size_mm_override)
    else:
        cols = int(rng.integers(8, 20))
        rows = int(rng.integers(6, 16))
        fill = float(rng.uniform(0.45, 0.85))
        board_w_mm = visible_w_mm * fill
        board_h_mm = visible_h_mm * fill
        square_size_mm = max(0.01, min(board_w_mm / cols, board_h_mm / rows))

    board: dict = {
        "type": "charuco",
        "square_size_mm": float(square_size_mm),
        "marker_size_mm": float(board_marker_size_mm_override)
        if board_marker_size_mm_override is not None
        else float(0.70 * square_size_mm),
        "squares_x": cols,
        "squares_y": rows,
        "cols": cols,
        "rows": rows,
        "aruco_dictionary": "DICT_4X4_1000",
        "pixels_per_square": int(board_pixels_per_square_override)
        if board_pixels_per_square_override is not None
        else 80,
    }

    texture = _make_board_texture(board, pattern)
    if texture is not None:
        board["_texture_img"] = texture
    board["texture_interp"] = str(tex_interp)

    baseline_mm = (
        float(baseline_mm_override)
        if baseline_mm_override is not None
        else float(rng.uniform(0.02, 0.25) * visible_w_mm)
    )

    dist_left, dist_right = _make_distortion(distort, float(distort_strength), rng)

    board_meta = {k: v for k, v in board.items() if not str(k).startswith("_")}

    meta = {
        "schema_version": "stereocomplex.dataset.v0",
        "stereo": stereo_meta,
        "board": board_meta,
        "sim_params": {
            "camera_model": "pinhole",
            "f_um": f_um,
            "baseline_mm": baseline_mm,
            "tz_schedule_mm": [float(x) for x in tz_schedule_mm] if tz_schedule_mm is not None else None,
            "tex_interp": str(tex_interp),
            "distortion_model": str(distort),
            "distortion_left": brown_to_dict(dist_left),
            "distortion_right": brown_to_dict(dist_right),
            "image_format": str(image_format),
            "outside_mask": str(outside_mask),
            "blur_fwhm_um": float(blur_fwhm_um),
            "blur_fwhm_px": float(blur_fwhm_px),
            "blur_edge_factor": float(blur_edge_factor),
            "blur_edge_start": float(blur_edge_start),
            "blur_edge_power": float(blur_edge_power),
            "noise_std": float(noise_std),
            "z_only_mode": bool(z_only_mode),
        },
    }
    (scene_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    view_meta = parse_view_meta(view)
    cam = PinholeCamera(f_um=f_um)

    # Left camera at origin; right camera shifted along +X in world (mm).
    oL = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    oR = np.array([baseline_mm, 0.0, 0.0], dtype=np.float64)

    # Prepare ray grids (pixel -> undistorted ray direction) for both cameras.
    d_cam_left = _ray_grid_from_pixels(view_meta, cam, dist_left)
    d_cam_right = _ray_grid_from_pixels(view_meta, cam, dist_right)

    # Fixed plane basis in world; per frame, random pose (rotation+translation).
    frames_path = scene_dir / "frames.jsonl"
    frames_f = frames_path.open("w", encoding="utf-8")

    all_frame_id: list[int] = []
    all_xyz: list[np.ndarray] = []
    all_uvL: list[np.ndarray] = []
    all_uvR: list[np.ndarray] = []

    all_corner_frame: list[np.ndarray] = []
    all_corner_id: list[np.ndarray] = []
    all_corner_xyz: list[np.ndarray] = []
    all_corner_uvL: list[np.ndarray] = []
    all_corner_uvR: list[np.ndarray] = []

    for frame_id in range(frames_per_scene):
        # Rejection-sample poses until the board yields enough valid GT points in both views.
        accepted = False
        max_attempts = 1 if z_only_mode else 25
        for _attempt in range(max_attempts):
            if tz_schedule_mm is not None:
                tz = float(tz_schedule_mm[frame_id])
            else:
                tz = float(tz_nominal * rng.uniform(0.7, 1.3))
            if z_only_mode:
                tx = 0.0
                ty = 0.0
                tilt_x = 0.0
                tilt_y = 0.0
            else:
                tx = float(rng.uniform(-0.15, 0.15) * visible_w_mm)
                ty = float(rng.uniform(-0.15, 0.15) * visible_h_mm)
                tilt_x = float(rng.uniform(-0.30, 0.30))
                tilt_y = float(rng.uniform(-0.30, 0.30))
            R = _rot_y(tilt_y) @ _rot_x(tilt_x)
            t = np.array([tx, ty, tz], dtype=np.float64)

            P = int(rng.integers(400, 1200))
            pts_plane = _sample_plane_points_mm(P, board, rng)
            xyz_world = (R @ np.c_[pts_plane, np.zeros(P)].T).T + t[None, :]
            uvL = _project_points_pinhole(view_meta, cam, dist_left, oL, xyz_world)
            uvR = _project_points_pinhole(view_meta, cam, dist_right, oR, xyz_world)
            valid = _in_image(uvL, width, height) & _in_image(uvR, width, height) & (xyz_world[:, 2] > 0)
            if int(np.count_nonzero(valid)) >= 80:
                xyz_world = xyz_world[valid]
                uvL = uvL[valid]
                uvR = uvR[valid]
                accepted = True
                break

        if not accepted:
            # Fallback: keep frame with whatever survived (possibly empty) to preserve determinism.
            xyz_world = np.zeros((0, 3), dtype=np.float64)
            uvL = np.zeros((0, 2), dtype=np.float64)
            uvR = np.zeros((0, 2), dtype=np.float64)
            R = np.eye(3, dtype=np.float64)
            t = np.array([0.0, 0.0, tz_nominal], dtype=np.float64)

        imgL = _render_plane_texture(
            d_cam_left,
            oL,
            R,
            t,
            board,
            view_meta,
            tex_interp,
            outside_mask,
            blur_fwhm_um,
            blur_fwhm_px,
            blur_edge_factor,
            blur_edge_start,
            blur_edge_power,
            noise_std,
            rng,
        )
        imgR = _render_plane_texture(
            d_cam_right,
            oR,
            R,
            t,
            board,
            view_meta,
            tex_interp,
            outside_mask,
            blur_fwhm_um,
            blur_fwhm_px,
            blur_edge_factor,
            blur_edge_start,
            blur_edge_power,
            noise_std,
            rng,
        )

        ext = _image_ext(image_format)
        left_name = f"{frame_id:06d}.{ext}"
        right_name = f"{frame_id:06d}.{ext}"
        _save_gray_image(left_dir / left_name, imgL, image_format=image_format)
        _save_gray_image(right_dir / right_name, imgR, image_format=image_format)

        frames_f.write(json.dumps({"frame_id": frame_id, "left": left_name, "right": right_name}) + "\n")

        all_frame_id.append(np.full((xyz_world.shape[0],), frame_id, dtype=np.int32))
        all_xyz.append(xyz_world.astype(np.float32))
        all_uvL.append(uvL.astype(np.float32))
        all_uvR.append(uvR.astype(np.float32))

        if board.get("type") == "charuco":
            corner_id, corners_plane = _charuco_inner_corners_mm(board)
            if corners_plane.size:
                xyz_c_world = (R @ np.c_[corners_plane, np.zeros(corners_plane.shape[0])].T).T + t[None, :]
                uvL_c = _project_points_pinhole(view_meta, cam, dist_left, oL, xyz_c_world)
                uvR_c = _project_points_pinhole(view_meta, cam, dist_right, oR, xyz_c_world)
                valid_c = _in_image(uvL_c, width, height) & _in_image(uvR_c, width, height) & (xyz_c_world[:, 2] > 0)
                all_corner_frame.append(np.full((int(np.count_nonzero(valid_c)),), frame_id, dtype=np.int32))
                all_corner_id.append(corner_id[valid_c].astype(np.int32))
                all_corner_xyz.append(xyz_c_world[valid_c].astype(np.float32))
                all_corner_uvL.append(uvL_c[valid_c].astype(np.float32))
                all_corner_uvR.append(uvR_c[valid_c].astype(np.float32))

    frames_f.close()

    frame_id_arr = np.concatenate(all_frame_id, axis=0) if all_frame_id else np.zeros((0,), np.int32)
    xyz_arr = np.concatenate(all_xyz, axis=0) if all_xyz else np.zeros((0, 3), np.float32)
    uvL_arr = np.concatenate(all_uvL, axis=0) if all_uvL else np.zeros((0, 2), np.float32)
    uvR_arr = np.concatenate(all_uvR, axis=0) if all_uvR else np.zeros((0, 2), np.float32)

    np.savez_compressed(
        scene_dir / "gt_points.npz",
        frame_id=frame_id_arr,
        XYZ_world_mm=xyz_arr,
        uv_left_px=uvL_arr,
        uv_right_px=uvR_arr,
    )

    if board.get("type") == "charuco":
        corner_frame = (
            np.concatenate(all_corner_frame, axis=0) if all_corner_frame else np.zeros((0,), np.int32)
        )
        corner_id = np.concatenate(all_corner_id, axis=0) if all_corner_id else np.zeros((0,), np.int32)
        corner_xyz = (
            np.concatenate(all_corner_xyz, axis=0) if all_corner_xyz else np.zeros((0, 3), np.float32)
        )
        corner_uvL = (
            np.concatenate(all_corner_uvL, axis=0) if all_corner_uvL else np.zeros((0, 2), np.float32)
        )
        corner_uvR = (
            np.concatenate(all_corner_uvR, axis=0) if all_corner_uvR else np.zeros((0, 2), np.float32)
        )
        np.savez_compressed(
            scene_dir / "gt_charuco_corners.npz",
            frame_id=corner_frame,
            corner_id=corner_id,
            XYZ_world_mm=corner_xyz,
            uv_left_px=corner_uvL,
            uv_right_px=corner_uvR,
        )


def _rot_x(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float64)


def _sample_plane_points_mm(n: int, board: dict, rng: np.random.Generator) -> np.ndarray:
    square = float(board["square_size_mm"])
    if "squares_x" in board and "squares_y" in board:
        cols = float(board["squares_x"])
        rows = float(board["squares_y"])
    else:
        cols = float(board["cols"])
        rows = float(board["rows"])
    w_mm = square * cols
    h_mm = square * rows
    x = rng.uniform(-0.5 * w_mm, 0.5 * w_mm, size=(n,))
    y = rng.uniform(-0.5 * h_mm, 0.5 * h_mm, size=(n,))
    return np.stack([x, y], axis=-1)


def _charuco_inner_corners_mm(board: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (corner_id, corners_xy_mm) for the inner chessboard corners of a Charuco board.
    Convention: board centered at (0,0), spans [-w/2, +w/2] in plane local coords.
    """
    if "squares_x" in board and "squares_y" in board:
        cols = int(board["squares_x"])
        rows = int(board["squares_y"])
    else:
        cols = int(board.get("cols", 0))
        rows = int(board.get("rows", 0))
    if cols < 2 or rows < 2:
        return np.zeros((0,), np.int32), np.zeros((0, 2), np.float64)

    square = float(board["square_size_mm"])
    w_mm = square * cols
    h_mm = square * rows

    xs = (-0.5 * w_mm + square * np.arange(1, cols, dtype=np.float64))
    ys = (-0.5 * h_mm + square * np.arange(1, rows, dtype=np.float64))
    xx, yy = np.meshgrid(xs, ys)
    corners = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)

    # Stable corner ids: row-major in the (rows-1, cols-1) grid.
    cid = np.arange(corners.shape[0], dtype=np.int32)
    return cid, corners


def _make_distortion(
    model: str, strength: float, rng: np.random.Generator
) -> tuple[BrownDistortion, BrownDistortion]:
    model = str(model)
    strength = float(strength)
    if model == "none" or strength <= 0.0:
        return BrownDistortion(), BrownDistortion()
    if model != "brown":
        raise ValueError("distort must be none|brown")

    # Coefficients are intentionally small; strength scales them.
    # Typical normalized radii are <= ~0.8 for most pixels.
    def sample_one() -> BrownDistortion:
        s = strength
        return BrownDistortion(
            k1=float(rng.uniform(-0.25, 0.25) * s),
            k2=float(rng.uniform(-0.15, 0.15) * s),
            p1=float(rng.uniform(-0.02, 0.02) * s),
            p2=float(rng.uniform(-0.02, 0.02) * s),
            k3=float(rng.uniform(-0.05, 0.05) * s),
        )

    return sample_one(), sample_one()


def _ray_grid_from_pixels(view_meta, cam: PinholeCamera, dist: BrownDistortion) -> np.ndarray:
    """
    Pixel -> sensor (µm) -> normalized distorted -> undistort -> ray directions.
    Returns (H,W,3) in camera/world frame (identity).
    """
    x_um, y_um = pixel_grid_um(view_meta)
    xd = x_um / float(cam.f_um)
    yd = y_um / float(cam.f_um)
    x, y = dist.undistort(xd, yd)
    return cam.ray_directions_cam_from_norm(x, y)


def _project_points_pinhole(
    view_meta, cam: PinholeCamera, dist: BrownDistortion, origin_mm: np.ndarray, xyz_world_mm: np.ndarray
) -> np.ndarray:
    # Camera frame == world frame in MVP; origin defines translation only.
    p_cam = xyz_world_mm - origin_mm[None, :]
    x_mm = p_cam[:, 0]
    y_mm = p_cam[:, 1]
    z_mm = p_cam[:, 2]
    x = x_mm / z_mm
    y = y_mm / z_mm
    xd, yd = dist.distort(x, y)
    x_um = xd * float(cam.f_um)
    y_um = yd * float(cam.f_um)
    u_px, v_px = sensor_um_to_pixel(view_meta, x_um, y_um)
    return np.stack([u_px, v_px], axis=-1)


def _in_image(uv: np.ndarray, w: int, h: int) -> np.ndarray:
    return (uv[:, 0] >= 0) & (uv[:, 0] <= (w - 1)) & (uv[:, 1] >= 0) & (uv[:, 1] <= (h - 1))


def _render_plane_texture(
    d_cam: np.ndarray,
    origin_mm: np.ndarray,
    R_plane: np.ndarray,
    t_plane_mm: np.ndarray,
    board: dict,
    view_meta,
    tex_interp: str,
    outside_mask: str,
    blur_fwhm_um: float,
    blur_fwhm_px: float,
    blur_edge_factor: float,
    blur_edge_start: float,
    blur_edge_power: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Vectorized ray-plane render of a simple grid texture on the plane.
    """
    # Plane: points satisfy n·(X - t) = 0, with n = R*[0,0,1].
    n = R_plane @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    denom = d_cam @ n
    denom = np.where(np.abs(denom) < 1e-9, np.nan, denom)
    t = ((t_plane_mm - origin_mm) @ n) / denom  # (H,W)
    X = origin_mm[None, None, :] + t[..., None] * d_cam  # (H,W,3)

    # Plane local coords: inverse rotation to plane frame.
    Xp = (R_plane.T @ (X - t_plane_mm[None, None, :]).reshape(-1, 3).T).T.reshape(X.shape)
    xp = Xp[..., 0]
    yp = Xp[..., 1]

    square = float(board["square_size_mm"])
    if "squares_x" in board and "squares_y" in board:
        cols = int(board["squares_x"])
        rows = int(board["squares_y"])
    else:
        cols = int(board["cols"])
        rows = int(board["rows"])
    w_mm = square * cols
    h_mm = square * rows

    finite = np.isfinite(t) & (t > 0)
    inside = (
        finite
        & (xp >= -0.5 * w_mm)
        & (xp <= 0.5 * w_mm)
        & (yp >= -0.5 * h_mm)
        & (yp <= 0.5 * h_mm)
    )
    tex = _sample_board_texture(board, xp, yp, inside, tex_interp=tex_interp).astype(np.float64) / 255.0

    # Background + illumination gradient + noise.
    bg = float(rng.uniform(0.05, 0.25))
    img = np.full(tex.shape, bg, dtype=np.float64)
    img[inside] = tex[inside]

    # Mild vignette/gradient.
    H, W = img.shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    grad = 1.0 + 0.15 * xx + 0.10 * yy
    img *= grad

    # Noise.
    img = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255.0 + 0.5).astype(np.uint8)

    sigma_px = float(0.0)
    if blur_fwhm_px > 0:
        sigma_px = fwhm_to_sigma(blur_fwhm_px)
    elif blur_fwhm_um > 0:
        sigma_um = fwhm_to_sigma(blur_fwhm_um)
        pitch_x_um = view_meta.sensor.pixel_pitch_um * view_meta.sensor.binning_xy[0]
        pitch_y_um = view_meta.sensor.pixel_pitch_um * view_meta.sensor.binning_xy[1]
        resize_x, resize_y = view_meta.preprocess.resize_xy
        sigma_x = sigma_um * resize_x / pitch_x_um
        sigma_y = sigma_um * resize_y / pitch_y_um
        img_u8 = gaussian_blur_edge_varying_u8(
            img_u8,
            sigma_x_center=sigma_x,
            sigma_y_center=sigma_y,
            edge_factor=float(blur_edge_factor),
            edge_start=float(blur_edge_start),
            edge_power=float(blur_edge_power),
        )
    if sigma_px > 0:
        img_u8 = gaussian_blur_edge_varying_u8(
            img_u8,
            sigma_x_center=sigma_px,
            sigma_y_center=sigma_px,
            edge_factor=float(blur_edge_factor),
            edge_start=float(blur_edge_start),
            edge_power=float(blur_edge_power),
        )

    if noise_std > 0:
        img_f = img_u8.astype(np.float32) / 255.0
        img_f += rng.normal(0.0, float(noise_std), size=img_f.shape).astype(np.float32)
        img_f = np.clip(img_f, 0.0, 1.0)
        img_u8 = (img_f * 255.0 + 0.5).astype(np.uint8)

    if outside_mask == "hard":
        img_u8[~inside] = 0
    elif outside_mask != "none":
        raise ValueError("outside_mask must be none|hard")

    return img_u8


def _image_ext(image_format: str) -> str:
    image_format = str(image_format).lower()
    if image_format == "png":
        return "png"
    if image_format == "webp":
        return "webp"
    raise ValueError("image_format must be png|webp")


def _save_gray_image(path: Path, img_u8: np.ndarray, image_format: str) -> None:
    image_format = str(image_format).lower()
    im = Image.fromarray(img_u8, mode="L")
    if image_format == "png":
        im.save(path)
        return
    if image_format == "webp":
        # Lossless WebP requires Pillow built with libwebp support.
        try:
            from PIL import features

            if not features.check("webp"):
                raise RuntimeError("Pillow has no WebP support (PIL.features.check('webp') is False).")
        except Exception:
            pass
        im.save(path, lossless=True, quality=100, method=6)
        return
    raise ValueError("image_format must be png|webp")


def _make_board_texture(board: dict, pattern: str) -> np.ndarray | None:
    """
    Generates and returns the board texture as a grayscale uint8 image, or None to use analytic grid.
    """
    if pattern not in ("auto", "charuco", "grid"):
        raise ValueError("pattern must be auto|charuco|grid")
    if pattern == "grid":
        board["type"] = "texture_grid"
        return None
    if board.get("type") != "charuco":
        return None
    try:
        spec = CharucoSpec(
            squares_x=int(board["squares_x"]),
            squares_y=int(board["squares_y"]),
            square_size_mm=float(board["square_size_mm"]),
            marker_size_mm=float(board["marker_size_mm"]),
            aruco_dictionary=str(board.get("aruco_dictionary", "DICT_4X4_1000")),
            pixels_per_square=int(board.get("pixels_per_square", 80)),
        )
        return generate_charuco_texture(spec)
    except Exception:
        if pattern == "charuco":
            raise
        board["type"] = "texture_grid"
        return None


def _sample_board_texture(
    board: dict, xp: np.ndarray, yp: np.ndarray, inside: np.ndarray, tex_interp: str
) -> np.ndarray:
    """
    Samples board texture as uint8 image intensity using plane local coords (mm).
    If no texture is available, uses an analytic grid pattern.
    """
    square = float(board["square_size_mm"])

    if "squares_x" in board and "squares_y" in board:
        cols = int(board["squares_x"])
        rows = int(board["squares_y"])
    else:
        cols = int(board.get("cols", 12))
        rows = int(board.get("rows", 9))

    w_mm = square * cols
    h_mm = square * rows

    tex = np.zeros_like(xp, dtype=np.uint8)

    if board.get("type") == "charuco":
        # Texture expected to be cached on the board dict by _render_plane_texture caller.
        img = board.get("_texture_img")
        if isinstance(img, np.ndarray) and img.ndim == 2:
            Ht, Wt = img.shape
            u = (xp + 0.5 * w_mm) / w_mm
            v = (yp + 0.5 * h_mm) / h_mm
            map_x = np.zeros_like(xp, dtype=np.float32)
            map_y = np.zeros_like(yp, dtype=np.float32)
            # Pixel-center convention: pixel centers at integer coordinates, borders at -0.5 and W-0.5.
            # This avoids a half-texel bias when mapping plane coordinates to the discrete texture.
            map_x[inside] = (u[inside] * Wt - 0.5).astype(np.float32)
            map_y[inside] = (v[inside] * Ht - 0.5).astype(np.float32)
            sampled = _remap_texture(img, map_x, map_y, interp=tex_interp)
            tex[inside] = sampled[inside]
            return tex

    # Analytic grid fallback.
    gx = np.floor((xp + 0.5 * w_mm) / square).astype(np.int32)
    gy = np.floor((yp + 0.5 * h_mm) / square).astype(np.int32)
    v = ((gx + gy) & 1).astype(np.uint8)
    tex[inside] = (0.2 * 255 + 0.7 * 255 * v[inside]).astype(np.uint8)
    return tex


def _remap_texture(img_u8: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, interp: str) -> np.ndarray:
    interp = str(interp)
    if interp not in ("nearest", "linear", "cubic", "lanczos4"):
        raise ValueError("tex_interp must be nearest|linear|cubic|lanczos4")

    try:
        import cv2  # type: ignore

        flags = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4,
        }[interp]
        out = cv2.remap(
            img_u8,
            map_x,
            map_y,
            interpolation=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return out
    except Exception:
        if interp == "nearest":
            xi = np.clip(np.rint(map_x).astype(np.int32), 0, img_u8.shape[1] - 1)
            yi = np.clip(np.rint(map_y).astype(np.int32), 0, img_u8.shape[0] - 1)
            return img_u8[yi, xi]
        return _remap_bilinear_numpy(img_u8, map_x, map_y)


def _remap_bilinear_numpy(img_u8: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    H, W = img_u8.shape
    x = np.clip(map_x.astype(np.float64), 0.0, W - 1.0)
    y = np.clip(map_y.astype(np.float64), 0.0, H - 1.0)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)

    wx = (x - x0).astype(np.float64)
    wy = (y - y0).astype(np.float64)

    Ia = img_u8[y0, x0].astype(np.float64)
    Ib = img_u8[y0, x1].astype(np.float64)
    Ic = img_u8[y1, x0].astype(np.float64)
    Id = img_u8[y1, x1].astype(np.float64)

    wa = (1.0 - wx) * (1.0 - wy)
    wb = wx * (1.0 - wy)
    wc = (1.0 - wx) * wy
    wd = wx * wy
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return np.clip(out + 0.5, 0.0, 255.0).astype(np.uint8)
