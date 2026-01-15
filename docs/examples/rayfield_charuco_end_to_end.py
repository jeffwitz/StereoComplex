"""
End-to-end example: OpenCV "raw" ChArUco corners vs. plane ray-field second pass.

This script is designed for documentation and teaching:
- takes stereo image pairs (left/right) from a dataset/v0 scene,
- computes "raw" 2D ChArUco corners using OpenCV's Charuco detector / interpolation,
- computes a non-parametric "plane ray-field" second pass from ArUco marker corners only,
- computes 2D errors against ground truth (synthetic datasets only),
- exports plots (ECDF, histograms) and optional overlays as PNGs.

It intentionally avoids matplotlib to keep dependencies minimal (plots are drawn with OpenCV).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


Side = Literal["left", "right"]


@dataclass(frozen=True)
class ErrorSummary:
    n: int
    rms: float
    p50: float
    p95: float
    max: float


@dataclass(frozen=True)
class FrameSideCache:
    frame_id: int
    side: Side
    obj_xy: np.ndarray  # (N,2) board coords of marker corners
    img_uv: np.ndarray  # (N,2) image coords of marker corners
    gt_uv: dict[int, np.ndarray]  # corner_id -> uv


def _quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def summarize_errors(errors: list[float]) -> ErrorSummary:
    e = np.asarray(errors, dtype=np.float64)
    if e.size == 0:
        return ErrorSummary(n=0, rms=float("nan"), p50=float("nan"), p95=float("nan"), max=float("nan"))
    return ErrorSummary(
        n=int(e.size),
        rms=float(np.sqrt(np.mean(e * e))),
        p50=_quantile(e, 0.50),
        p95=_quantile(e, 0.95),
        max=float(np.max(e)),
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frames(scene_dir: Path) -> list[dict[str, Any]]:
    frames_path = scene_dir / "frames.jsonl"
    frames: list[dict[str, Any]] = []
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        frames.append(json.loads(line))
    return frames


def build_charuco_board_from_meta(meta: dict[str, Any]):
    import cv2  # type: ignore
    import cv2.aruco as aruco  # type: ignore

    board_meta = meta["board"]
    if board_meta.get("type") != "charuco":
        raise ValueError("This example expects board.type == 'charuco'")

    dict_name = str(board_meta.get("aruco_dictionary", "DICT_4X4_1000"))
    dict_id = getattr(aruco, dict_name, None)
    if dict_id is None:
        raise ValueError(f"Unknown aruco_dictionary: {dict_name}")
    dictionary = aruco.getPredefinedDictionary(dict_id)

    squares_x = int(board_meta["squares_x"])
    squares_y = int(board_meta["squares_y"])
    square_size = float(board_meta["square_size_mm"])
    marker_size = float(board_meta["marker_size_mm"])

    if hasattr(aruco, "CharucoBoard"):
        board = aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, dictionary)
    elif hasattr(aruco, "CharucoBoard_create"):  # pragma: no cover
        board = aruco.CharucoBoard_create(squares_x, squares_y, square_size, marker_size, dictionary)
    else:  # pragma: no cover
        raise RuntimeError("cv2.aruco does not expose CharucoBoard APIs in this build.")

    detector_params = aruco.DetectorParameters()
    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 50
    detector_params.cornerRefinementMinAccuracy = 1e-3

    charuco_detector = None
    if hasattr(aruco, "CharucoDetector"):
        charuco_detector = aruco.CharucoDetector(board)
        if hasattr(charuco_detector, "setDetectorParameters"):
            charuco_detector.setDetectorParameters(detector_params)

    aruco_detector = None
    if charuco_detector is None and hasattr(aruco, "ArucoDetector"):
        aruco_detector = aruco.ArucoDetector(dictionary, detector_params)

    return cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector


def detect_markers_and_charuco(
    cv2,
    aruco,
    dictionary,
    board,
    detector_params,
    aruco_detector,
    charuco_detector,
    img: np.ndarray,
):
    """
    Returns:
      charuco_corners_px, charuco_ids, marker_corners_px, marker_ids

    Coordinate convention:
      - marker corners are already consistent with this repo's dataset pixel-center convention,
      - OpenCV Charuco corners are typically shifted by +0.5 px; we apply -0.5 correction here.
    """
    if charuco_detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)
    else:
        if aruco_detector is not None:
            marker_corners, marker_ids, _rejected = aruco_detector.detectMarkers(img)
        else:  # pragma: no cover
            marker_corners, marker_ids, _rejected = aruco.detectMarkers(img, dictionary, parameters=detector_params)

        charuco_corners, charuco_ids = None, None
        if hasattr(aruco, "interpolateCornersCharuco") and marker_ids is not None and len(marker_ids) > 0:
            ret = aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, board)
            if ret is not None:
                # OpenCV Python bindings vary across versions; accept both common shapes.
                if len(ret) == 3:
                    charuco_corners, charuco_ids, _ = ret
                elif len(ret) == 4:  # pragma: no cover
                    _, charuco_corners, charuco_ids, _ = ret

    # Normalize outputs.
    if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
        marker_ids_arr = np.zeros((0,), dtype=np.int32)
        marker_corners_arr: list[np.ndarray] = []
    else:
        marker_ids_arr = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
        marker_corners_arr = [np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in marker_corners]

    if charuco_ids is None or charuco_corners is None or len(charuco_ids) == 0:
        charuco_ids_arr = np.zeros((0,), dtype=np.int32)
        charuco_xy = np.zeros((0, 2), dtype=np.float64)
    else:
        charuco_ids_arr = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        charuco_xy = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
        # Match dataset pixel-center convention (see docs/CONVENTIONS.md).
        charuco_xy = charuco_xy - 0.5

    return charuco_xy, charuco_ids_arr, marker_corners_arr, marker_ids_arr


def _gt_map_for_scene(scene_dir: Path) -> dict[int, dict[Side, dict[int, np.ndarray]]]:
    """
    Returns: gt[frame_id][side][corner_id] = uv (2,)
    """
    gt_path = scene_dir / "gt_charuco_corners.npz"
    data = np.load(str(gt_path))
    frame_id = np.asarray(data["frame_id"], dtype=np.int32).reshape(-1)
    corner_id = np.asarray(data["corner_id"], dtype=np.int32).reshape(-1)
    uv_left = np.asarray(data["uv_left_px"], dtype=np.float64).reshape(-1, 2)
    uv_right = np.asarray(data["uv_right_px"], dtype=np.float64).reshape(-1, 2)

    gt: dict[int, dict[Side, dict[int, np.ndarray]]] = {}
    for fid in np.unique(frame_id).tolist():
        mask = frame_id == int(fid)
        ids = corner_id[mask]
        l = uv_left[mask]
        r = uv_right[mask]
        gt[int(fid)] = {
            "left": {int(i): l[k] for k, i in enumerate(ids.tolist())},
            "right": {int(i): r[k] for k, i in enumerate(ids.tolist())},
        }
    return gt


def compute_errors(pred_xy: np.ndarray, pred_ids: np.ndarray, gt_uv_by_id: dict[int, np.ndarray]) -> list[float]:
    errors: list[float] = []
    for pid, pxy in zip(pred_ids.tolist(), pred_xy.tolist(), strict=True):
        gt = gt_uv_by_id.get(int(pid))
        if gt is None:
            continue
        dx = float(pxy[0] - gt[0])
        dy = float(pxy[1] - gt[1])
        errors.append(float(np.hypot(dx, dy)))
    return errors


def draw_ecdf_plot(
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    title: str,
    out_path: Path,
    x_max: float | None = None,
) -> None:
    import cv2  # type: ignore

    w, h = 1100, 700
    pad_l, pad_r, pad_t, pad_b = 80, 30, 60, 70
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Axis box.
    x0, y0 = pad_l, h - pad_b
    x1, y1 = w - pad_r, pad_t
    cv2.rectangle(img, (x0, y1), (x1, y0), (0, 0, 0), 2)
    cv2.putText(img, title, (pad_l, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    # Determine x range (robust).
    all_e = np.concatenate([e for _, e, _ in series if e.size > 0], axis=0) if any(e.size > 0 for _, e, _ in series) else np.zeros((0,))
    if x_max is None:
        x_max = float(np.quantile(all_e, 0.995)) if all_e.size > 0 else 1.0
        x_max = max(0.25, x_max)

    # Ticks.
    for t in np.linspace(0.0, x_max, 6):
        xx = int(x0 + (x1 - x0) * (t / x_max))
        cv2.line(img, (xx, y0), (xx, y0 + 5), (0, 0, 0), 1)
        cv2.putText(img, f"{t:.2f}", (xx - 15, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for t in np.linspace(0.0, 1.0, 6):
        yy = int(y0 - (y0 - y1) * t)
        cv2.line(img, (x0 - 5, yy), (x0, yy), (0, 0, 0), 1)
        cv2.putText(img, f"{t:.1f}", (x0 - 45, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "error (px)", (w // 2 - 40, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "ECDF", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Plot series.
    legend_y = y1 + 15
    for k, (name, e, color) in enumerate(series):
        if e.size == 0:
            continue
        e_sorted = np.sort(e)
        y = (np.arange(e_sorted.size, dtype=np.float64) + 1.0) / float(e_sorted.size)
        pts = []
        for ex, ey in zip(e_sorted.tolist(), y.tolist(), strict=True):
            ex = float(min(ex, x_max))
            px = int(x0 + (x1 - x0) * (ex / x_max))
            py = int(y0 - (y0 - y1) * ey)
            pts.append((px, py))
        for p, q in zip(pts[:-1], pts[1:], strict=True):
            cv2.line(img, p, q, color, 2, cv2.LINE_AA)
        # Legend.
        lx = x0 + 10 + 260 * k
        cv2.rectangle(img, (lx, legend_y), (lx + 20, legend_y + 12), color, -1)
        cv2.putText(img, name, (lx + 28, legend_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def draw_hist_plot(
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    title: str,
    out_path: Path,
    x_max: float | None = None,
    bins: int = 40,
) -> None:
    import cv2  # type: ignore

    w, h = 1100, 700
    pad_l, pad_r, pad_t, pad_b = 80, 30, 60, 70
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    x0, y0 = pad_l, h - pad_b
    x1, y1 = w - pad_r, pad_t
    cv2.rectangle(img, (x0, y1), (x1, y0), (0, 0, 0), 2)
    cv2.putText(img, title, (pad_l, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    all_e = np.concatenate([e for _, e, _ in series if e.size > 0], axis=0) if any(e.size > 0 for _, e, _ in series) else np.zeros((0,))
    if x_max is None:
        x_max = float(np.quantile(all_e, 0.995)) if all_e.size > 0 else 1.0
        x_max = max(0.25, x_max)

    # Compute histograms.
    edges = np.linspace(0.0, x_max, int(bins) + 1, dtype=np.float64)
    hists = []
    ymax = 1.0
    for _, e, _ in series:
        if e.size == 0:
            hists.append(np.zeros((bins,), dtype=np.float64))
            continue
        e2 = np.clip(e, 0.0, x_max)
        counts, _ = np.histogram(e2, bins=edges)
        hists.append(counts.astype(np.float64))
        ymax = max(ymax, float(np.max(counts)))

    # Plot bars (side-by-side with slight offset).
    bin_w_px = (x1 - x0) / float(bins)
    for si, (name, _e, color) in enumerate(series):
        counts = hists[si]
        for b in range(bins):
            c = float(counts[b])
            if c <= 0:
                continue
            # bar geometry
            bx0 = x0 + int(b * bin_w_px + 0.1 * bin_w_px + si * 0.35 * bin_w_px)
            bx1 = x0 + int(b * bin_w_px + 0.1 * bin_w_px + (si + 1) * 0.35 * bin_w_px)
            by1 = y0
            by0 = int(y0 - (y0 - y1) * (c / ymax))
            cv2.rectangle(img, (bx0, by0), (bx1, by1), color, -1)

    # Axes labels/ticks.
    for t in np.linspace(0.0, x_max, 6):
        xx = int(x0 + (x1 - x0) * (t / x_max))
        cv2.line(img, (xx, y0), (xx, y0 + 5), (0, 0, 0), 1)
        cv2.putText(img, f"{t:.2f}", (xx - 15, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "error (px)", (w // 2 - 40, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "count", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Legend.
    legend_y = y1 + 15
    for k, (name, _e, color) in enumerate(series):
        lx = x0 + 10 + 260 * k
        cv2.rectangle(img, (lx, legend_y), (lx + 20, legend_y + 12), color, -1)
        cv2.putText(img, name, (lx + 28, legend_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def draw_overlay(
    img_gray: np.ndarray,
    gt_uv: dict[int, np.ndarray],
    pred_raw: tuple[np.ndarray, np.ndarray] | None,
    pred_ray: tuple[np.ndarray, np.ndarray] | None,
    out_path: Path,
    max_points: int = 250,
    overlay_scale: int = 6,
    margin_px: int = 40,
    vector_scale: float = 10.0,
    min_vector_len_px: float = 0.25,
) -> None:
    """
    Overlay (GT vs predictions) for quick visual inspection.

    Notes:
      - errors can be sub-pixel; to make differences visible, we crop around the board and upscale the crop.
    """
    import cv2  # type: ignore

    overlay_scale = int(max(1, overlay_scale))
    margin_px = int(max(0, margin_px))
    vector_scale = float(max(1.0, vector_scale))
    min_vector_len_px = float(max(0.0, min_vector_len_px))

    # Compute a crop around the region of interest (GT + predictions) to keep the overlay readable.
    pts = []
    for _cid, uv in list(gt_uv.items())[:max_points]:
        pts.append(np.asarray(uv, dtype=np.float64))
    if pred_raw is not None:
        xy, ids = pred_raw
        for cid, p in zip(ids.tolist(), xy.tolist(), strict=True):
            if int(cid) in gt_uv:
                pts.append(np.asarray(p, dtype=np.float64))
                if min_vector_len_px > 0:
                    gt0 = np.asarray(gt_uv[int(cid)], dtype=np.float64)
                    pr0 = np.asarray(p, dtype=np.float64)
                    pts.append(gt0 + vector_scale * (pr0 - gt0))
    if pred_ray is not None:
        xy, ids = pred_ray
        for cid, p in zip(ids.tolist(), xy.tolist(), strict=True):
            if int(cid) in gt_uv:
                pts.append(np.asarray(p, dtype=np.float64))
                if min_vector_len_px > 0:
                    gt0 = np.asarray(gt_uv[int(cid)], dtype=np.float64)
                    pr0 = np.asarray(p, dtype=np.float64)
                    pts.append(gt0 + vector_scale * (pr0 - gt0))

    h0, w0 = img_gray.shape[:2]
    if pts:
        P = np.stack(pts, axis=0)
        xmin = int(max(0, np.floor(float(np.min(P[:, 0])) - margin_px)))
        xmax = int(min(w0 - 1, np.ceil(float(np.max(P[:, 0])) + margin_px)))
        ymin = int(max(0, np.floor(float(np.min(P[:, 1])) - margin_px)))
        ymax = int(min(h0 - 1, np.ceil(float(np.max(P[:, 1])) + margin_px)))
        if xmax <= xmin or ymax <= ymin:
            xmin, ymin, xmax, ymax = 0, 0, w0 - 1, h0 - 1
    else:
        xmin, ymin, xmax, ymax = 0, 0, w0 - 1, h0 - 1

    crop = img_gray[ymin : ymax + 1, xmin : xmax + 1]
    crop = cv2.resize(crop, (crop.shape[1] * overlay_scale, crop.shape[0] * overlay_scale), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "GT (green) vs OpenCV raw (red) vs ray-field (blue)"
    subtitle = f"Overlay: crop×{overlay_scale}, vectors ×{vector_scale:g} when |e|<{min_vector_len_px:g}px"
    cv2.putText(img, title, (20, 30), font, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, title, (20, 30), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, subtitle, (20, 55), font, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, subtitle, (20, 55), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    def to_xy(uv: np.ndarray) -> tuple[int, int]:
        x = (float(uv[0]) - xmin) * overlay_scale
        y = (float(uv[1]) - ymin) * overlay_scale
        return int(round(x)), int(round(y))

    radius = max(6, int(round(1.5 * overlay_scale)))
    thick = max(1, int(round(overlay_scale / 3)))

    def draw_cross(pt: tuple[int, int], color: tuple[int, int, int], r: int) -> None:
        x, y = pt
        cv2.line(img, (x - r, y), (x + r, y), color, thick, cv2.LINE_AA)
        cv2.line(img, (x, y - r), (x, y + r), color, thick, cv2.LINE_AA)

    def draw_circle(pt: tuple[int, int], color: tuple[int, int, int], r: int) -> None:
        cv2.circle(img, pt, r, color, thick, cv2.LINE_AA)

    def draw_arrow(p0: tuple[int, int], p1: tuple[int, int], color: tuple[int, int, int]) -> None:
        # arrowedLine is more readable than a simple segment for tiny residuals.
        cv2.arrowedLine(img, p0, p1, color, thick, cv2.LINE_AA, tipLength=0.25)

    def draw_pred(
        pred: tuple[np.ndarray, np.ndarray] | None,
        color: tuple[int, int, int],
        circle_r: int,
    ) -> None:
        if pred is None:
            return
        xy, ids = pred
        for cid, p in zip(ids.tolist(), xy.tolist(), strict=True):
            cid = int(cid)
            if cid not in gt_uv:
                continue
            gt0 = np.asarray(gt_uv[cid], dtype=np.float64)
            pr0 = np.asarray(p, dtype=np.float64)
            gt_pt = to_xy(gt0)
            pr_pt = to_xy(pr0)

            # Always draw the true predicted location as a circle.
            draw_circle(pr_pt, color, r=circle_r)

            # Draw a residual arrow; if too small, magnify direction for visibility.
            d0 = pr0 - gt0
            n0 = float(np.hypot(float(d0[0]), float(d0[1])))
            if n0 <= 1e-12:
                continue

            if min_vector_len_px > 0.0 and n0 < min_vector_len_px:
                vis0 = gt0 + vector_scale * d0
                vis_pt = to_xy(vis0)
                # faint segment to the true prediction
                cv2.line(img, gt_pt, pr_pt, (80, 80, 80), 1, cv2.LINE_AA)
                draw_arrow(gt_pt, vis_pt, color)
            else:
                draw_arrow(gt_pt, pr_pt, color)

    # Draw predictions first (so GT can be drawn on top with an outline).
    draw_pred(pred_raw, (0, 0, 255), circle_r=radius + 2)
    draw_pred(pred_ray, (255, 0, 0), circle_r=radius)

    # GT last with black outline for readability.
    gt_subset = list(gt_uv.items())[:max_points]
    for _cid, uv in gt_subset:
        pt = to_xy(np.asarray(uv, dtype=np.float64))
        draw_cross(pt, (0, 0, 0), r=radius + 2)
        draw_cross(pt, (0, 255, 0), r=radius)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def _predict_map(pred: tuple[np.ndarray, np.ndarray] | None) -> dict[int, np.ndarray]:
    if pred is None:
        return {}
    xy, ids = pred
    return {int(i): np.asarray(p, dtype=np.float64) for i, p in zip(ids.tolist(), xy.tolist(), strict=True)}


def _safe_crop_with_padding(img: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """
    Returns an (h,w) crop; out-of-bounds areas are padded with zeros.
    """
    import cv2  # type: ignore

    ih, iw = img.shape[:2]
    x1 = x0 + w
    y1 = y0 + h

    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(iw, x1)
    sy1 = min(ih, y1)

    crop = img[sy0:sy1, sx0:sx1]
    top = sy0 - y0
    left = sx0 - x0
    bottom = y1 - sy1
    right = x1 - sx1
    if top or left or bottom or right:
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    return crop


def draw_micro_overlay_pair(
    img_gray: np.ndarray,
    gt_uv: dict[int, np.ndarray],
    pred_raw: tuple[np.ndarray, np.ndarray] | None,
    pred_ray: tuple[np.ndarray, np.ndarray] | None,
    out_path: Path,
    corner_id: int,
    radius_px: int,
    micro_scale: int = 80,
) -> None:
    """
    Micro overlay around one ChArUco corner: two panels (raw vs ray-field).

    The crop covers a (2*radius_px+1)x(2*radius_px+1) neighborhood in the original image,
    then is upscaled by `micro_scale` so that sub-pixel deltas become visible.
    """
    import cv2  # type: ignore

    micro_scale = int(max(8, micro_scale))
    radius_px = int(max(1, radius_px))

    gt0 = gt_uv.get(int(corner_id))
    if gt0 is None:
        return
    gt0 = np.asarray(gt0, dtype=np.float64)

    # Crop origin aligned to integer pixel grid for a readable neighborhood.
    ix = int(np.floor(float(gt0[0])))
    iy = int(np.floor(float(gt0[1])))
    x0 = ix - radius_px
    y0 = iy - radius_px
    w = 2 * radius_px + 1
    h = 2 * radius_px + 1

    crop = _safe_crop_with_padding(img_gray, x0, y0, w, h)
    crop_up = cv2.resize(crop, (w * micro_scale, h * micro_scale), interpolation=cv2.INTER_NEAREST)
    crop_up = cv2.cvtColor(crop_up, cv2.COLOR_GRAY2BGR)

    def uv_to_xy(uv: np.ndarray) -> tuple[int, int]:
        x = (float(uv[0]) - float(x0)) * micro_scale
        y = (float(uv[1]) - float(y0)) * micro_scale
        return int(round(x)), int(round(y))

    def draw_pixel_grid(img: np.ndarray) -> None:
        gh, gw = img.shape[:2]
        for k in range(0, gw + 1, micro_scale):
            cv2.line(img, (k, 0), (k, gh - 1), (60, 60, 60), 1, cv2.LINE_AA)
        for k in range(0, gh + 1, micro_scale):
            cv2.line(img, (0, k), (gw - 1, k), (60, 60, 60), 1, cv2.LINE_AA)

    def draw_point_set(img: np.ndarray, title: str, pred_uv: np.ndarray | None, color: tuple[int, int, int]) -> None:
        draw_pixel_grid(img)

        # GT cross (with outline for contrast)
        gt_pt = uv_to_xy(gt0)
        for c, r, t in [((0, 0, 0), 12, 4), ((0, 255, 0), 10, 2)]:
            cv2.line(img, (gt_pt[0] - r, gt_pt[1]), (gt_pt[0] + r, gt_pt[1]), c, t, cv2.LINE_AA)
            cv2.line(img, (gt_pt[0], gt_pt[1] - r), (gt_pt[0], gt_pt[1] + r), c, t, cv2.LINE_AA)

        if pred_uv is not None:
            pr_pt = uv_to_xy(pred_uv)
            cv2.arrowedLine(img, gt_pt, pr_pt, color, 2, cv2.LINE_AA, tipLength=0.25)
            cv2.circle(img, pr_pt, 12, color, 2, cv2.LINE_AA)

        # Title
        cv2.putText(img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 1, cv2.LINE_AA)

    raw_map = _predict_map(pred_raw)
    ray_map = _predict_map(pred_ray)
    raw_uv = raw_map.get(int(corner_id))
    ray_uv = ray_map.get(int(corner_id))

    left = crop_up.copy()
    right = crop_up.copy()
    draw_point_set(left, f"raw (id={corner_id})", raw_uv, (0, 0, 255))
    draw_point_set(right, f"ray-field (id={corner_id})", ray_uv, (255, 0, 0))

    # Stack panels with a thin separator.
    sep = np.full((left.shape[0], 8, 3), 0, dtype=np.uint8)
    out = np.concatenate([left, sep, right], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


def _pick_best_improvement_corner_id(
    gt_uv: dict[int, np.ndarray],
    pred_raw: tuple[np.ndarray, np.ndarray] | None,
    pred_ray: tuple[np.ndarray, np.ndarray] | None,
) -> int | None:
    raw_map = _predict_map(pred_raw)
    ray_map = _predict_map(pred_ray)
    best_id: int | None = None
    best_gain = -1e18
    for cid, gt0 in gt_uv.items():
        pr = raw_map.get(int(cid))
        py = ray_map.get(int(cid))
        if pr is None or py is None:
            continue
        gt0 = np.asarray(gt0, dtype=np.float64)
        e_raw = float(np.hypot(float(pr[0] - gt0[0]), float(pr[1] - gt0[1])))
        e_ray = float(np.hypot(float(py[0] - gt0[0]), float(py[1] - gt0[1])))
        gain = e_raw - e_ray
        if gain > best_gain:
            best_gain = gain
            best_id = int(cid)
    return best_id


def _pick_image_corner_id(gt_uv: dict[int, np.ndarray]) -> int | None:
    if not gt_uv:
        return None
    # "Corner of the board" in image coordinates: smallest (x+y) -> top-left.
    items = [(int(cid), float(uv[0]) + float(uv[1])) for cid, uv in gt_uv.items()]
    items.sort(key=lambda t: t[1])
    return int(items[0][0])


def _parse_float_list(s: str) -> list[float]:
    vals = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("expected a non-empty comma-separated float list")
    return vals


def draw_lambda_sweep_plot(
    lam: list[float],
    rms: list[float],
    p95: list[float],
    title: str,
    out_path: Path,
) -> None:
    """
    Draw RMS and P95 vs lambda, with x-axis in log10(lambda).
    """
    import cv2  # type: ignore

    if len(lam) != len(rms) or len(lam) != len(p95) or len(lam) == 0:
        return

    x = np.log10(np.asarray(lam, dtype=np.float64))
    y1 = np.asarray(rms, dtype=np.float64)
    y2 = np.asarray(p95, dtype=np.float64)

    w, h = 1100, 700
    pad_l, pad_r, pad_t, pad_b = 90, 30, 60, 80
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    x0, y0 = pad_l, h - pad_b
    x1p, y1p = w - pad_r, pad_t
    cv2.rectangle(img, (x0, y1p), (x1p, y0), (0, 0, 0), 2)
    cv2.putText(img, title, (pad_l, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(np.concatenate([y1, y2]))), float(np.max(np.concatenate([y1, y2])))
    ymin = max(0.0, ymin - 0.05 * (ymax - ymin + 1e-9))
    ymax = ymax + 0.10 * (ymax - ymin + 1e-9)
    if abs(xmax - xmin) < 1e-9:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0

    def to_px(xx: float, yy: float) -> tuple[int, int]:
        px = int(x0 + (x1p - x0) * ((xx - xmin) / (xmax - xmin)))
        py = int(y0 - (y0 - y1p) * ((yy - ymin) / (ymax - ymin)))
        return px, py

    # X ticks (label with lambda values).
    for k, l in enumerate(lam):
        xx = float(np.log10(float(l)))
        px, _ = to_px(xx, ymin)
        cv2.line(img, (px, y0), (px, y0 + 6), (0, 0, 0), 1)
        cv2.putText(img, f"{l:g}", (px - 18, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Y ticks.
    for t in np.linspace(ymin, ymax, 6):
        _, py = to_px(xmin, float(t))
        cv2.line(img, (x0 - 6, py), (x0, py), (0, 0, 0), 1)
        cv2.putText(img, f"{t:.3f}", (x0 - 75, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(img, "lambda (TPS)", (w // 2 - 60, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "error (px)", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Series.
    col_rms = (0, 0, 255)  # red
    col_p95 = (0, 160, 0)  # green
    pts_rms = [to_px(float(x[i]), float(y1[i])) for i in range(len(lam))]
    pts_p95 = [to_px(float(x[i]), float(y2[i])) for i in range(len(lam))]
    for p, q in zip(pts_rms[:-1], pts_rms[1:], strict=True):
        cv2.line(img, p, q, col_rms, 2, cv2.LINE_AA)
    for p, q in zip(pts_p95[:-1], pts_p95[1:], strict=True):
        cv2.line(img, p, q, col_p95, 2, cv2.LINE_AA)
    for p in pts_rms:
        cv2.circle(img, p, 5, col_rms, -1, cv2.LINE_AA)
    for p in pts_p95:
        cv2.circle(img, p, 5, col_p95, -1, cv2.LINE_AA)

    # Legend.
    legend_y = y1p + 15
    cv2.rectangle(img, (x0 + 10, legend_y), (x0 + 30, legend_y + 12), col_rms, -1)
    cv2.putText(img, "RMS", (x0 + 38, legend_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0 + 120, legend_y), (x0 + 140, legend_y + 12), col_p95, -1)
    cv2.putText(img, "P95", (x0 + 148, legend_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def draw_residual_amplitude_heatmap(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    board_xy: np.ndarray,
    pred_map_fn,
    out_path: Path,
    grid_nx: int = 220,
    grid_ny: int = 140,
    viz_scale: int = 4,
    title_suffix: str = "",
) -> None:
    """
    Visualize ||r(x,y)|| on the board plane for one frame/view.

    We compute a base homography H (RANSAC), then evaluate the mapping on a dense grid
    of (x,y) on the board. The residual is defined as:
        r(x,y) = u_pred(x,y) - pi(H [x,y,1]^T)
    and we plot ||r|| as a heatmap (in pixels).

    Preferred rendering is via matplotlib + viridis (publication-friendly). If matplotlib
    is not available, we fall back to an OpenCV colormap.
    """
    import cv2  # type: ignore

    viz_scale = int(max(1, viz_scale))

    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    board_xy = np.asarray(board_xy, dtype=np.float64)

    H, _mask = cv2.findHomography(obj_xy, img_uv, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        return

    def proj(Hh: np.ndarray, pts: np.ndarray) -> np.ndarray:
        ph = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
        uvw = (Hh @ ph.T).T
        return uvw[:, :2] / (uvw[:, 2:3] + 1e-12)

    xmin, xmax = float(np.min(board_xy[:, 0])), float(np.max(board_xy[:, 0]))
    ymin, ymax = float(np.min(board_xy[:, 1])), float(np.max(board_xy[:, 1]))
    xs = np.linspace(xmin, xmax, int(max(10, grid_nx)), dtype=np.float64)
    ys = np.linspace(ymin, ymax, int(max(10, grid_ny)), dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    query = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)

    base = proj(H, query)
    pred = np.asarray(pred_map_fn(query), dtype=np.float64).reshape(-1, 2)
    res = pred - base
    amp = np.sqrt(np.sum(res * res, axis=1)).reshape(Y.shape[0], X.shape[1])

    vmax = float(np.quantile(amp, 0.995))
    vmax = max(1e-6, vmax)
    title = "Residual amplitude on board plane: ||r(x,y)|| (px)"
    if str(title_suffix).strip():
        title = f"{title} {title_suffix}"
    subtitle = f"clipped at q99.5={vmax:.3f}px"

    # Preferred: matplotlib + viridis.
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        fig_w = max(6.5, float(grid_nx) / 35.0)
        fig_h = max(3.5, float(grid_ny) / 35.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
        im = ax.imshow(
            amp,
            cmap="viridis",
            origin="lower",
            vmin=0.0,
            vmax=vmax,
            extent=(xmin, xmax, ymin, ymax),
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xlabel("x (board)")
        ax.set_ylabel("y (board)")
        ax.set_title(f"{title}\n{subtitle}")
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("||r|| (px)")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return
    except Exception:
        # Fallback: OpenCV colormap if matplotlib isn't installed.
        pass

    img_u8 = np.clip((amp / vmax) * 255.0, 0.0, 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(img_u8, cv2.COLORMAP_TURBO)
    if viz_scale > 1:
        heat = cv2.resize(heat, (heat.shape[1] * viz_scale, heat.shape[0] * viz_scale), interpolation=cv2.INTER_NEAREST)

    # Build a vertical colorbar.
    bar_w = 60
    bar_h = heat.shape[0]
    grad = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(np.repeat(grad, bar_w, axis=1), cv2.COLORMAP_TURBO)

    # Annotate.
    top = 95
    left = 30
    canvas = np.full((heat.shape[0] + top + 25, heat.shape[1] + bar_w + 90, 3), 255, dtype=np.uint8)
    canvas[top : top + heat.shape[0], left : left + heat.shape[1]] = heat
    canvas[top : top + bar.shape[0], left + heat.shape[1] + 25 : left + heat.shape[1] + 25 + bar.shape[1]] = bar

    cv2.putText(canvas, title, (left, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (left, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    # Colorbar labels (0, vmax/2, vmax).
    for frac, lab in [(0.0, "0"), (0.5, f"{0.5*vmax:.3f}"), (1.0, f"{vmax:.3f}")]:
        y = int(top + (1.0 - frac) * (bar_h - 1))
        x = left + heat.shape[1] + 25 + bar_w + 10
        cv2.putText(canvas, lab, (x, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path, help="Path to dataset root, e.g. dataset/v0_png")
    ap.add_argument("--split", default="train", help="Dataset split (train|val|test).")
    ap.add_argument("--scene", default="scene_0000", help="Scene directory name inside split.")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit number of frames (0 = all).")
    ap.add_argument("--out", type=Path, default=Path("docs/assets/rayfield_worked_example/_out"), help="Output directory.")
    ap.add_argument("--save-overlays", action="store_true", help="Save overlay PNGs for the first frame.")
    ap.add_argument("--overlay-scale", type=int, default=6, help="Overlay crop upscaling factor (for subpixel visibility).")
    ap.add_argument("--overlay-margin", type=int, default=40, help="Overlay crop margin (px) around detected board area.")
    ap.add_argument(
        "--overlay-vector-scale",
        type=float,
        default=10.0,
        help="When residuals are tiny, magnify vectors for visibility (purely for overlay).",
    )
    ap.add_argument(
        "--overlay-min-vector-len",
        type=float,
        default=0.25,
        help="Residual length threshold (px) below which overlay vectors are magnified.",
    )
    ap.add_argument("--micro-scale", type=int, default=80, help="Upscaling factor for micro-overlays.")
    ap.add_argument("--micro-radius", type=int, default=3, help="Neighborhood radius (px) for the micro-overlay point.")
    ap.add_argument("--micro-corner-radius", type=int, default=1, help="Neighborhood radius (px) for the micro-overlay board corner.")
    ap.add_argument("--grid-nx", type=int, default=16)
    ap.add_argument("--grid-ny", type=int, default=10)
    ap.add_argument("--smooth-lambda", type=float, default=3.0)
    ap.add_argument("--huber-c", type=float, default=3.0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument(
        "--rayfield-backend",
        choices=("grid", "tps"),
        default="tps",
        help="Residual model for the 2nd pass: grid (Laplacian-regularized) or TPS (smoothing spline).",
    )
    ap.add_argument("--tps-lam", type=float, default=10.0, help="TPS smoothing parameter (only if --rayfield-backend=tps).")
    ap.add_argument(
        "--sweep-tps-lam",
        type=str,
        default="",
        help="Optional: comma-separated list of TPS lambda values to sweep; produces additional plots.",
    )
    args = ap.parse_args()

    dataset_root: Path = args.dataset_root
    scene_dir = dataset_root / str(args.split) / str(args.scene)
    meta = load_json(scene_dir / "meta.json")

    cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector = build_charuco_board_from_meta(meta)

    # Import ray-field implementation from the project (this is what we want to document).
    from stereocomplex.eval.charuco_detection import (  # noqa: PLC0415
        _predict_points_rayfield,
        _predict_points_rayfield_tps_robust,
    )

    gt = _gt_map_for_scene(scene_dir)
    frames = load_frames(scene_dir)
    if args.max_frames and args.max_frames > 0:
        frames = frames[: int(args.max_frames)]

    chess = np.asarray(board.getChessboardCorners(), dtype=np.float64)[:, :2]
    all_ids = np.arange(chess.shape[0], dtype=np.int32)

    errors = {
        "left": {"raw": [], "rayfield": []},
        "right": {"raw": [], "rayfield": []},
    }

    # Precompute marker-id -> board-plane (x,y) corners.
    board_ids = np.asarray(board.getIds(), dtype=np.int32).reshape(-1)
    board_obj = board.getObjPoints()
    id_to_obj2 = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

    caches: list[FrameSideCache] = []

    for fi, frame in enumerate(frames):
        frame_id = int(frame["frame_id"])
        for side in ("left", "right"):
            img_path = scene_dir / side / str(frame[side])
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")

            raw_xy, raw_ids, marker_corners, marker_ids = detect_markers_and_charuco(
                cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector, img
            )

            gt_uv = gt.get(frame_id, {}).get(side, {})
            if not gt_uv:
                continue

            # 1) OpenCV raw ChArUco errors.
            raw_err = compute_errors(raw_xy, raw_ids, gt_uv)
            errors[side]["raw"].extend(raw_err)

            # 2) Ray-field second pass (markers only).
            obj_pts = []
            img_pts = []
            for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
                o = id_to_obj2.get(int(mid))
                if o is None or mc.shape != (4, 2) or o.shape != (4, 2):
                    continue
                obj_pts.append(o)
                img_pts.append(mc)
            if obj_pts:
                obj_xy = np.concatenate(obj_pts, axis=0)
                img_uv = np.concatenate(img_pts, axis=0)
                caches.append(FrameSideCache(frame_id=frame_id, side=side, obj_xy=obj_xy, img_uv=img_uv, gt_uv=gt_uv))

                if str(args.rayfield_backend) == "tps":
                    ray_xy = _predict_points_rayfield_tps_robust(obj_xy, img_uv, chess, lam=float(args.tps_lam))
                else:
                    ray_xy = _predict_points_rayfield(
                        obj_xy,
                        img_uv,
                        chess,
                        grid_size=(int(args.grid_nx), int(args.grid_ny)),
                        smooth_lambda=float(args.smooth_lambda),
                        huber_c=float(args.huber_c),
                        iters=int(args.iters),
                    )
                ray_err = compute_errors(ray_xy, all_ids, gt_uv)
                errors[side]["rayfield"].extend(ray_err)
            else:
                ray_xy = np.zeros((0, 2), dtype=np.float64)

            # Save overlays for the first frame only (both views).
            if args.save_overlays and fi == 0:
                overlay_path = args.out / "overlays" / f"{side}_frame{frame_id:06d}.png"
                pred_raw = (raw_xy, raw_ids) if raw_xy.size > 0 else None
                pred_ray = (ray_xy, all_ids) if ray_xy.size > 0 else None
                draw_overlay(
                    img,
                    gt_uv,
                    pred_raw,
                    pred_ray,
                    overlay_path,
                    overlay_scale=int(args.overlay_scale),
                    margin_px=int(args.overlay_margin),
                    vector_scale=float(args.overlay_vector_scale),
                    min_vector_len_px=float(args.overlay_min_vector_len),
                )

                # Micro overlays: (1) one point + ~2–3 px neighborhood, (2) one board corner + ~1 px neighborhood.
                best_id = _pick_best_improvement_corner_id(gt_uv, pred_raw, pred_ray)
                if best_id is not None:
                    draw_micro_overlay_pair(
                        img,
                        gt_uv,
                        pred_raw,
                        pred_ray,
                        args.out / "micro_overlays" / f"{side}_best_frame{frame_id:06d}.png",
                        corner_id=int(best_id),
                        radius_px=int(args.micro_radius),
                        micro_scale=int(args.micro_scale),
                    )

                corner_id = _pick_image_corner_id(gt_uv)
                if corner_id is not None:
                    draw_micro_overlay_pair(
                        img,
                        gt_uv,
                        pred_raw,
                        pred_ray,
                        args.out / "micro_overlays" / f"{side}_corner_frame{frame_id:06d}.png",
                        corner_id=int(corner_id),
                        radius_px=int(args.micro_corner_radius),
                        micro_scale=int(args.micro_scale),
                    )

                # Residual amplitude field (visualizing aberrations) for this frame/view.
                if str(args.rayfield_backend) == "tps":
                    pred_fn = lambda q: _predict_points_rayfield_tps_robust(obj_xy, img_uv, q, lam=float(args.tps_lam))  # noqa: E731
                else:
                    pred_fn = lambda q: _predict_points_rayfield(  # noqa: E731
                        obj_xy,
                        img_uv,
                        q,
                        grid_size=(int(args.grid_nx), int(args.grid_ny)),
                        smooth_lambda=float(args.smooth_lambda),
                        huber_c=float(args.huber_c),
                        iters=int(args.iters),
                    )

                draw_residual_amplitude_heatmap(
                    obj_xy=obj_xy,
                    img_uv=img_uv,
                    board_xy=chess,
                    pred_map_fn=pred_fn,
                    out_path=args.out / "plots" / f"residual_amp_{side}_frame{frame_id:06d}.png",
                    viz_scale=4,
                    title_suffix=f"({side}, lam={float(args.tps_lam):g})" if str(args.rayfield_backend) == "tps" else f"({side})",
                )

    # Summaries.
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"rayfield_backend": str(args.rayfield_backend), "tps_lam": float(args.tps_lam)}
    for side in ("left", "right"):
        s_raw = summarize_errors(errors[side]["raw"])
        s_ray = summarize_errors(errors[side]["rayfield"])
        summary[side] = {"raw_charuco": s_raw.__dict__, "rayfield": s_ray.__dict__}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # Plots.
    colors = {"raw": (0, 0, 255), "rayfield": (255, 0, 0)}  # BGR
    ray_label = "ray-field (TPS)" if str(args.rayfield_backend) == "tps" else "ray-field (grid)"
    for side in ("left", "right"):
        e_raw = np.asarray(errors[side]["raw"], dtype=np.float64)
        e_ray = np.asarray(errors[side]["rayfield"], dtype=np.float64)
        draw_ecdf_plot(
            [
                ("OpenCV raw", e_raw, colors["raw"]),
                (ray_label, e_ray, colors["rayfield"]),
            ],
            title=f"ECDF error ({side})",
            out_path=out_dir / "plots" / f"ecdf_{side}.png",
        )
        draw_hist_plot(
            [
                ("OpenCV raw", e_raw, colors["raw"]),
                (ray_label, e_ray, colors["rayfield"]),
            ],
            title=f"Histogram error ({side})",
            out_path=out_dir / "plots" / f"hist_{side}.png",
        )

    print("Wrote:", out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))

    # Optional: sweep TPS lambda sensitivity (uses the cached correspondences).
    if str(args.sweep_tps_lam).strip():
        lam_list = _parse_float_list(str(args.sweep_tps_lam))
        lam_list = sorted(set(float(l) for l in lam_list))
        sweep = {"lam": lam_list, "left": [], "right": []}

        for lam in lam_list:
            for side in ("left", "right"):
                e: list[float] = []
                for item in caches:
                    if item.side != side:
                        continue
                    pred = _predict_points_rayfield_tps_robust(item.obj_xy, item.img_uv, chess, lam=float(lam))
                    e.extend(compute_errors(pred, all_ids, item.gt_uv))
                s = summarize_errors(e)
                sweep[side].append({"n": s.n, "rms": s.rms, "p95": s.p95})

        (out_dir / "tps_lambda_sweep.json").write_text(json.dumps(sweep, indent=2, sort_keys=True), encoding="utf-8")
        for side in ("left", "right"):
            rms = [float(d["rms"]) for d in sweep[side]]
            p95 = [float(d["p95"]) for d in sweep[side]]
            draw_lambda_sweep_plot(
                lam_list,
                rms,
                p95,
                title=f"TPS lambda sweep ({side})",
                out_path=out_dir / "plots" / f"tps_lambda_sweep_{side}.png",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
