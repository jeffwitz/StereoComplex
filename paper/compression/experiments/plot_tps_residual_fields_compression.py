#!/usr/bin/env python3
"""
Plot TPS residual fields under different codec settings.

This script is intended to generate a lightweight, reproducible qualitative figure for the paper:
the magnitude ||g(x,y)|| (px) of the TPS residual field learned on the board plane after a robust
homography fit.

It runs ArUco detection on a small subset of frames for two codec variants and visualizes the
average residual magnitude over that subset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

from stereocomplex.core.rayfield2d import _predict_points_tps_irls


def _load_meta(dataset_root: Path, codec: str) -> dict:
    meta_path = dataset_root / codec / "train" / "scene_0000" / "meta.json"
    return json.loads(meta_path.read_text())


def _build_board(meta: dict) -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.Dictionary]:
    aruco = cv2.aruco
    b = meta["board"]
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, b["aruco_dictionary"]))
    board = aruco.CharucoBoard(
        (int(b["squares_x"]), int(b["squares_y"])),
        float(b["square_size_mm"]),
        float(b["marker_size_mm"]),
        dictionary,
    )
    return board, dictionary


def _codec_frame_paths(scene_dir: Path, view: str, frame_ids: list[int]) -> list[Path]:
    view_dir = scene_dir / view
    paths = []
    for frame_id in frame_ids:
        stem = f"{frame_id:06d}"
        candidates = list(view_dir.glob(stem + ".*"))
        if not candidates:
            raise FileNotFoundError(f"Missing frame {stem} in {view_dir}")
        paths.append(candidates[0])
    return paths


def _detect_marker_corners(
    img_gray: np.ndarray, *, dictionary: cv2.aruco.Dictionary
) -> tuple[list[np.ndarray], np.ndarray]:
    aruco = cv2.aruco
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)
    corners, ids, _rejected = detector.detectMarkers(img_gray)
    if ids is None:
        return [], np.empty((0,), dtype=np.int32)
    return corners, ids.reshape(-1).astype(np.int32)


def _fit_residual_field(
    board: cv2.aruco.CharucoBoard,
    dictionary: cv2.aruco.Dictionary,
    img_paths: list[Path],
    *,
    lam: float,
    huber_c: float,
    iters: int,
    ransac_reproj_px: float,
    grid_nx: int,
    grid_ny: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    ids_board = board.getIds().reshape(-1).astype(np.int32)
    obj_board = board.getObjPoints()
    obj_by_id = {int(mid): np.asarray(obj_board[i], dtype=np.float64) for i, mid in enumerate(ids_board)}

    # Grid on the board plane in millimeters.
    w_mm, h_mm, _ = board.getRightBottomCorner()
    xs = np.linspace(0.0, float(w_mm), int(grid_nx), dtype=np.float64)
    ys = np.linspace(0.0, float(h_mm), int(grid_ny), dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    grid_xy = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)

    mags = []
    aruco = cv2.aruco
    for p in img_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {p}")

        corners, ids = _detect_marker_corners(img, dictionary=dictionary)
        if ids.size == 0:
            continue

        obj_xy = []
        img_uv = []
        for c, mid in zip(corners, ids, strict=False):
            if int(mid) not in obj_by_id:
                continue
            obj_pts = obj_by_id[int(mid)][:, :2]  # (4,2)
            img_pts = np.asarray(c, dtype=np.float64).reshape(-1, 2)  # (4,2)
            obj_xy.append(obj_pts)
            img_uv.append(img_pts)

        if not obj_xy:
            continue
        obj_xy = np.concatenate(obj_xy, axis=0)
        img_uv = np.concatenate(img_uv, axis=0)

        H, _mask = cv2.findHomography(
            obj_xy, img_uv, method=cv2.RANSAC, ransacReprojThreshold=float(ransac_reproj_px)
        )
        if H is None:
            H, _mask = cv2.findHomography(obj_xy, img_uv, method=0)
        if H is None:
            continue

        ph = np.concatenate([obj_xy, np.ones((obj_xy.shape[0], 1), dtype=np.float64)], axis=1)
        uvw = (H @ ph.T).T
        base_obs = uvw[:, :2] / (uvw[:, 2:3] + 1e-12)
        res_obs = img_uv - base_obs

        res_grid = _predict_points_tps_irls(
            obj_xy,
            res_obs,
            grid_xy,
            lam=float(lam),
            huber_c=float(huber_c),
            iters=int(iters),
        )
        mag = np.sqrt(np.sum(res_grid**2, axis=1)).reshape(grid_ny, grid_nx)
        mags.append(mag)

    if not mags:
        raise RuntimeError("No valid frames with detected markers; cannot plot TPS residual field.")

    mag_mean = np.mean(np.stack(mags, axis=0), axis=0)
    extent = (0.0, float(w_mm), 0.0, float(h_mm))
    return mag_mean, extent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/compression_sweep"))
    parser.add_argument("--scene", type=str, default="scene_0000")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--view", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--codec-a", type=str, default="png_lossless")
    parser.add_argument("--codec-b", type=str, default="webp_q70")
    parser.add_argument("--frame-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tps-lam", type=float, default=10.0)
    parser.add_argument("--huber-c", type=float, default=3.0)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--ransac-reproj-px", type=float, default=3.0)
    parser.add_argument("--grid-nx", type=int, default=240)
    parser.add_argument("--grid-ny", type=int, default=160)
    parser.add_argument("--out", type=Path, default=Path("paper/figures/tps_residual_fields_png_webp.png"))
    args = parser.parse_args()

    # Use board definition from codec A (all codecs share board geometry for the sweep).
    meta = _load_meta(args.dataset_root, args.codec_a)
    board, dictionary = _build_board(meta)

    scene_a = args.dataset_root / args.codec_a / args.split / args.scene
    scene_b = args.dataset_root / args.codec_b / args.split / args.scene
    paths_a = _codec_frame_paths(scene_a, args.view, args.frame_ids)
    paths_b = _codec_frame_paths(scene_b, args.view, args.frame_ids)

    mag_a, extent = _fit_residual_field(
        board,
        dictionary,
        paths_a,
        lam=args.tps_lam,
        huber_c=args.huber_c,
        iters=args.iters,
        ransac_reproj_px=args.ransac_reproj_px,
        grid_nx=args.grid_nx,
        grid_ny=args.grid_ny,
    )
    mag_b, _extent = _fit_residual_field(
        board,
        dictionary,
        paths_b,
        lam=args.tps_lam,
        huber_c=args.huber_c,
        iters=args.iters,
        ransac_reproj_px=args.ransac_reproj_px,
        grid_nx=args.grid_nx,
        grid_ny=args.grid_ny,
    )

    vmax = float(np.quantile(np.concatenate([mag_a.ravel(), mag_b.ravel()]), 0.995))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
    for ax, mag, title in [
        (axes[0], mag_a, f"{args.codec_a}"),
        (axes[1], mag_b, f"{args.codec_b}"),
    ]:
        im = ax.imshow(
            mag,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("x (board, mm)")
        ax.set_ylabel("y (board, mm)")
        ax.grid(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label(r"$\|\mathbf{g}(x,y)\|$ (px)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

