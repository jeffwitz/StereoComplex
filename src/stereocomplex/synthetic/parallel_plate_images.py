from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from stereocomplex.api.calibration import CharucoBoardSpec, detect_charuco_corners
from stereocomplex.api.corner_refinement import RefineMethod, refine_charuco_corners
from stereocomplex.sim.cpu.effects import fwhm_to_sigma, gaussian_blur_edge_varying_u8
from stereocomplex.sim.patterns.charuco import CharucoSpec, generate_charuco_texture
from stereocomplex.synthetic.parallel_plate import (
    SyntheticStereoDataset,
    parallel_plate_ray_from_pixel,
)


@dataclass(frozen=True)
class ParallelPlateImageRenderParams:
    """Photometric parameters for rendered non-central ChArUco images."""

    pixels_per_square: int = 90
    texture_interp: str = "linear"
    blur_fwhm_px: float = 0.6
    blur_edge_factor: float = 1.4
    blur_edge_start: float = 0.55
    blur_edge_power: float = 2.0
    noise_std: float = 0.002
    vignette_strength: float = 0.24
    illumination_gradient_x: float = 0.08
    illumination_gradient_y: float = 0.05
    background_level: float = 0.18
    image_format: str = "png"
    seed: int = 1234


@dataclass(frozen=True)
class RenderedParallelPlateImageDataset:
    """Rendered image counterpart of a synthetic parallel-plate observation set."""

    dataset: SyntheticStereoDataset
    board: CharucoBoardSpec
    left_images: list[Path]
    right_images: list[Path]
    render_params: ParallelPlateImageRenderParams


def charuco_inner_corners_object_points(board: CharucoBoardSpec) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ChArUco inner-corner ids and board-frame 3D points.

    The convention matches the CPU dataset generator: the board is centered at
    `(0,0)` and lies in the `z=0` plane.
    """
    cols = int(board.squares_x)
    rows = int(board.squares_y)
    if cols < 2 or rows < 2:
        raise ValueError("a ChArUco board needs at least 2x2 squares")
    square = float(board.square_size_mm)
    w_mm = square * cols
    h_mm = square * rows
    xs = -0.5 * w_mm + square * np.arange(1, cols, dtype=np.float64)
    ys = -0.5 * h_mm + square * np.arange(1, rows, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size, dtype=np.float64)])
    ids = np.arange(points.shape[0], dtype=np.int32)
    return ids, points


def render_parallel_plate_charuco_images(
    dataset: SyntheticStereoDataset,
    board: CharucoBoardSpec,
    out_dir: str | Path,
    params: ParallelPlateImageRenderParams | None = None,
) -> RenderedParallelPlateImageDataset:
    """
    Render stereo ChArUco images from the non-central parallel-plate oracle.

    Rendering is done by ray tracing each camera pixel through the oracle
    rayfield, intersecting the calibration board plane, then sampling the
    ChArUco texture. Vignetting, spatially varying blur and sensor noise are
    applied after geometric rendering.
    """
    if dataset.oracle_left_params is None or dataset.oracle_right_params is None:
        raise ValueError("dataset must keep oracle parallel-plate parameters")
    render_params = params or ParallelPlateImageRenderParams()
    out = Path(out_dir)
    left_dir = out / "left"
    right_dir = out / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    texture = _charuco_texture(board, render_params)
    rng = np.random.default_rng(int(render_params.seed))
    left_paths: list[Path] = []
    right_paths: list[Path] = []
    for frame_id, T_board_world in enumerate(dataset.board_poses):
        T_left_board = np.asarray(dataset.T_left_world, dtype=np.float64).reshape(4, 4) @ np.asarray(
            T_board_world, dtype=np.float64
        ).reshape(4, 4)
        T_right_board = np.asarray(dataset.T_right_world, dtype=np.float64).reshape(4, 4) @ np.asarray(
            T_board_world, dtype=np.float64
        ).reshape(4, 4)
        img_left = _render_one_view(
            image_size=dataset.image_size,
            K=dataset.K_left,
            plate_params=dataset.oracle_left_params,
            T_camera_board=T_left_board,
            board=board,
            texture=texture,
            params=render_params,
            rng=rng,
        )
        img_right = _render_one_view(
            image_size=dataset.image_size,
            K=dataset.K_right,
            plate_params=dataset.oracle_right_params,
            T_camera_board=T_right_board,
            board=board,
            texture=texture,
            params=render_params,
            rng=rng,
        )
        ext = _image_ext(render_params.image_format)
        left_path = left_dir / f"{frame_id:06d}.{ext}"
        right_path = right_dir / f"{frame_id:06d}.{ext}"
        _save_gray(left_path, img_left, render_params.image_format)
        _save_gray(right_path, img_right, render_params.image_format)
        left_paths.append(left_path)
        right_paths.append(right_path)

    return RenderedParallelPlateImageDataset(
        dataset=dataset,
        board=board,
        left_images=left_paths,
        right_images=right_paths,
        render_params=render_params,
    )


def detected_observations_from_rendered_parallel_plate(
    rendered: RenderedParallelPlateImageDataset,
    min_common_corners: int = 12,
    method2d: RefineMethod = "raw",
) -> SyntheticStereoDataset:
    """
    Detect ChArUco corners in rendered images and return BA-ready observations.

    Each frame contributes its own detected corner subset. Frames with different
    board poses (including poses where part of the board is outside the image)
    naturally see different corners. ``per_frame_object_points`` carries the
    per-frame 3D point arrays so the BA handles variable-size observation sets.
    """
    all_ids, all_object_points = charuco_inner_corners_object_points(rendered.board)
    object_by_id = {int(cid): all_object_points[i] for i, cid in enumerate(all_ids.tolist())}

    per_frame_obj: list[np.ndarray] = []
    per_frame_left: list[np.ndarray] = []
    per_frame_right: list[np.ndarray] = []

    for left_path, right_path in zip(rendered.left_images, rendered.right_images, strict=True):
        det_left = detect_charuco_corners(image=left_path, board=rendered.board)
        det_right = detect_charuco_corners(image=right_path, board=rendered.board)
        if det_left is None or det_right is None:
            raise RuntimeError(f"ChArUco detection failed for pair {left_path.name}")
        left_xy = refine_charuco_corners(method=method2d, board=rendered.board, detections=det_left)
        right_xy = refine_charuco_corners(method=method2d, board=rendered.board, detections=det_right)
        left_map = {int(cid): left_xy[i] for i, cid in enumerate(det_left.charuco_ids.tolist())}
        right_map = {int(cid): right_xy[i] for i, cid in enumerate(det_right.charuco_ids.tolist())}
        common = sorted(set(left_map).intersection(right_map).intersection(object_by_id))
        if len(common) < int(min_common_corners):
            raise RuntimeError(f"only {len(common)} common ChArUco corners in frame {left_path.name}")
        per_frame_obj.append(np.stack([object_by_id[cid] for cid in common]))
        per_frame_left.append(np.stack([left_map[cid] for cid in common]).astype(np.float64))
        per_frame_right.append(np.stack([right_map[cid] for cid in common]).astype(np.float64))

    base = rendered.dataset
    return SyntheticStereoDataset(
        object_points=all_object_points,
        board_poses=base.board_poses,
        left_pixels=per_frame_left,
        right_pixels=per_frame_right,
        per_frame_object_points=per_frame_obj,
        K_left=base.K_left,
        K_right=base.K_right,
        T_left_world=base.T_left_world,
        T_right_world=base.T_right_world,
        image_size=base.image_size,
        oracle_left_params=base.oracle_left_params,
        oracle_right_params=base.oracle_right_params,
    )


def _charuco_texture(board: CharucoBoardSpec, params: ParallelPlateImageRenderParams) -> np.ndarray:
    spec = CharucoSpec(
        squares_x=int(board.squares_x),
        squares_y=int(board.squares_y),
        square_size_mm=float(board.square_size_mm),
        marker_size_mm=float(board.marker_size_mm),
        aruco_dictionary=str(board.aruco_dictionary),
        pixels_per_square=int(params.pixels_per_square),
    )
    return generate_charuco_texture(spec)


def _render_one_view(
    *,
    image_size: tuple[int, int],
    K: np.ndarray,
    plate_params,
    T_camera_board: np.ndarray,
    board: CharucoBoardSpec,
    texture: np.ndarray,
    params: ParallelPlateImageRenderParams,
    rng: np.random.Generator,
) -> np.ndarray:
    width, height = int(image_size[0]), int(image_size[1])
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij")
    origins, directions = parallel_plate_ray_from_pixel(xx.reshape(-1), yy.reshape(-1), K, plate_params)
    origins = origins.reshape(height, width, 3)
    directions = directions.reshape(height, width, 3)

    T = np.asarray(T_camera_board, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    n = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    denom = np.sum(directions * n.reshape(1, 1, 3), axis=2)
    denom = np.where(np.abs(denom) < 1e-10, np.nan, denom)
    lam = np.sum((t.reshape(1, 1, 3) - origins) * n.reshape(1, 1, 3), axis=2) / denom
    X = origins + lam[..., None] * directions
    Xp = (R.T @ (X - t.reshape(1, 1, 3)).reshape(-1, 3).T).T.reshape(height, width, 3)
    xp = Xp[..., 0]
    yp = Xp[..., 1]

    w_mm = float(board.squares_x) * float(board.square_size_mm)
    h_mm = float(board.squares_y) * float(board.square_size_mm)
    inside = (
        np.isfinite(lam)
        & (lam > 0.0)
        & (xp >= -0.5 * w_mm)
        & (xp <= 0.5 * w_mm)
        & (yp >= -0.5 * h_mm)
        & (yp <= 0.5 * h_mm)
    )
    sampled = _sample_texture(texture, xp, yp, inside, w_mm, h_mm, params.texture_interp).astype(np.float64) / 255.0
    img = np.full((height, width), float(params.background_level), dtype=np.float64)
    img[inside] = sampled[inside]

    xnorm = (xx - 0.5 * (width - 1)) / max(1.0, 0.5 * (width - 1))
    ynorm = (yy - 0.5 * (height - 1)) / max(1.0, 0.5 * (height - 1))
    r2 = np.clip(xnorm * xnorm + ynorm * ynorm, 0.0, 2.0) / 2.0
    vignette = np.clip(1.0 - float(params.vignette_strength) * r2, 0.05, 1.0)
    gradient = 1.0 + float(params.illumination_gradient_x) * xnorm + float(params.illumination_gradient_y) * ynorm
    img = np.clip(img * vignette * gradient, 0.0, 1.0)
    img_u8 = (img * 255.0 + 0.5).astype(np.uint8)

    if params.blur_fwhm_px > 0.0:
        img_u8 = gaussian_blur_edge_varying_u8(
            img_u8,
            sigma_x_center=fwhm_to_sigma(params.blur_fwhm_px),
            sigma_y_center=fwhm_to_sigma(params.blur_fwhm_px),
            edge_factor=float(params.blur_edge_factor),
            edge_start=float(params.blur_edge_start),
            edge_power=float(params.blur_edge_power),
        )
    if params.noise_std > 0.0:
        img_f = img_u8.astype(np.float32) / 255.0
        img_f += rng.normal(0.0, float(params.noise_std), size=img_f.shape).astype(np.float32)
        img_u8 = (np.clip(img_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return img_u8


def _sample_texture(
    texture: np.ndarray,
    xp: np.ndarray,
    yp: np.ndarray,
    inside: np.ndarray,
    board_w_mm: float,
    board_h_mm: float,
    interp: str,
) -> np.ndarray:
    height, width = texture.shape
    map_x = np.zeros_like(xp, dtype=np.float32)
    map_y = np.zeros_like(yp, dtype=np.float32)
    u = (xp + 0.5 * float(board_w_mm)) / float(board_w_mm)
    v = (yp + 0.5 * float(board_h_mm)) / float(board_h_mm)
    map_x[inside] = (u[inside] * width - 0.5).astype(np.float32)
    map_y[inside] = (v[inside] * height - 0.5).astype(np.float32)
    try:
        import cv2  # type: ignore

        flags = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4,
        }[str(interp)]
        return cv2.remap(texture, map_x, map_y, flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    except Exception:
        xi = np.clip(np.rint(map_x).astype(np.int32), 0, width - 1)
        yi = np.clip(np.rint(map_y).astype(np.int32), 0, height - 1)
        return texture[yi, xi]


def _image_ext(image_format: str) -> str:
    fmt = str(image_format).lower()
    if fmt == "png":
        return "png"
    if fmt == "webp":
        return "webp"
    raise ValueError("image_format must be png|webp")


def _save_gray(path: Path, img_u8: np.ndarray, image_format: str) -> None:
    fmt = str(image_format).lower()
    im = Image.fromarray(img_u8, mode="L")
    if fmt == "png":
        im.save(path)
    elif fmt == "webp":
        im.save(path, lossless=True, quality=100, method=6)
    else:
        raise ValueError("image_format must be png|webp")
