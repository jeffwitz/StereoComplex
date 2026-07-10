#!/usr/bin/env python3
"""Validate the CMO marker-TPS and Hessian corner preprocessing on real images."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cmo_corner_preprocessing import (  # noqa: E402
    abs_det_hessian,
    complete_corners_hessian,
    ids_to_grid,
    marker_tps_corners,
    otsu_mask,
    second_tps_pass,
)

PYCASO = ROOT / "examples" / "pycaso_data" / "Exemple" / "Images_example"
IMAGE_DIRS = {
    "left": PYCASO / "left_calibration11",
    "right": PYCASO / "right_calibration11",
}
OUT = ROOT / "docs" / "assets" / "pycaso_real_data" / "corner_preprocessing_validation.json"
NCX, NCY, SQUARE_MM = 16, 12, 0.3
MASK_FRACTIONS = (0.10, 0.20, 0.30, 0.50)
RNG_SEED = 20260710


def _detectors():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard(
        (NCX, NCY), SQUARE_MM, SQUARE_MM / 2, dictionary
    )
    board.setLegacyPattern(True)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 75
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.005
    params.maxMarkerPerimeterRate = 0.20
    params.polygonalApproxAccuracyRate = 0.08
    params.minCornerDistanceRate = 0.02
    params.minDistanceToBorder = 1
    return (
        board,
        aruco.ArucoDetector(dictionary, params),
        aruco.CharucoDetector(board),
    )


def _summary(errors: list[np.ndarray], ids: list[np.ndarray]) -> dict[str, object]:
    vectors = np.concatenate(errors, axis=0)
    corner_ids = np.concatenate(ids, axis=0)
    radial = np.linalg.norm(vectors, axis=1)
    bias = np.mean(vectors, axis=0)
    centered_radial = np.linalg.norm(vectors - bias, axis=1)
    grid = ids_to_grid(corner_ids, NCX)
    spatial_rms: list[list[float | None]] = []
    for gy in range(3):
        row: list[float | None] = []
        for gx in range(3):
            selected = (
                (grid[:, 0] >= gx * (NCX - 1) / 3)
                & (grid[:, 0] < (gx + 1) * (NCX - 1) / 3)
                & (grid[:, 1] >= gy * (NCY - 1) / 3)
                & (grid[:, 1] < (gy + 1) * (NCY - 1) / 3)
            )
            row.append(
                float(np.sqrt(np.mean(radial[selected] ** 2)))
                if np.any(selected)
                else None
            )
        spatial_rms.append(row)
    return {
        "n": int(radial.size),
        "bias_xy_px": bias.tolist(),
        "bias_norm_px": float(np.linalg.norm(bias)),
        "rms_px": float(np.sqrt(np.mean(radial**2))),
        "centered_rms_px": float(np.sqrt(np.mean(centered_radial**2))),
        "median_px": float(np.median(radial)),
        "p95_px": float(np.quantile(radial, 0.95)),
        "spatial_rms_3x3_px": spatial_rms,
    }


def main() -> int:
    board, marker_detector, charuco_detector = _detectors()
    chessboard_xy = np.asarray(board.getChessboardCorners(), dtype=np.float64)[:, :2]
    marker_object_xy = {
        int(marker_id): np.asarray(points, dtype=np.float64)[:, :2]
        for marker_id, points in zip(
            np.asarray(board.getIds()).reshape(-1),
            board.getObjPoints(),
            strict=True,
        )
    }
    rng = np.random.default_rng(RNG_SEED)
    errors: dict[str, list[np.ndarray]] = {
        "marker_tps": [],
        "double_tps": [],
        **{f"hessian_mask_{int(100 * f)}": [] for f in MASK_FRACTIONS},
    }
    error_ids: dict[str, list[np.ndarray]] = {key: [] for key in errors}
    views: list[dict[str, object]] = []

    for channel, image_dir in IMAGE_DIRS.items():
        for image_path in sorted(image_dir.glob("*.png"), key=lambda p: float(p.stem)):
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
            marker_corners, marker_ids, _ = marker_detector.detectMarkers(gray)
            if charuco_ids is None or len(charuco_ids) < 20:
                continue
            direct_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
            direct_xy = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
            first_tps = marker_tps_corners(
                marker_corners, marker_ids, marker_object_xy, chessboard_xy
            )
            if first_tps is None:
                continue
            final_tps = second_tps_pass(chessboard_xy, first_tps)
            errors["marker_tps"].append(first_tps[direct_ids] - direct_xy)
            errors["double_tps"].append(final_tps[direct_ids] - direct_xy)
            error_ids["marker_tps"].append(direct_ids)
            error_ids["double_tps"].append(direct_ids)

            response_mask = otsu_mask(abs_det_hessian(gray))
            for fraction in MASK_FRACTIONS:
                key = f"hessian_mask_{int(100 * fraction)}"
                n_mask = max(1, round(fraction * len(direct_ids)))
                held_positions = np.sort(
                    rng.choice(len(direct_ids), size=n_mask, replace=False)
                )
                kept = np.ones(len(direct_ids), dtype=bool)
                kept[held_positions] = False
                completed = complete_corners_hessian(
                    gray,
                    direct_xy[kept],
                    direct_ids[kept],
                    NCX,
                    NCY,
                    marker_object_xy=marker_object_xy,
                    chessboard_xy=chessboard_xy,
                    marker_corners=marker_corners,
                    marker_ids=marker_ids,
                    hessian_mask=response_mask,
                )
                held_ids = direct_ids[held_positions]
                errors[key].append(completed[held_ids] - direct_xy[held_positions])
                error_ids[key].append(held_ids)
            views.append({
                "channel": channel,
                "z": float(image_path.stem),
                "direct_charuco_corners": len(direct_ids),
                "aruco_markers": int(0 if marker_ids is None else len(marker_ids)),
            })

    result = {
        "method": (
            "Direct ChArUco detections are pseudo-ground truth. Marker-TPS uses "
            "the explicit ArUco-to-ChArUco half-pixel registration and is evaluated "
            "on every directly detected corner; Hessian completion is evaluated "
            "after deterministic random masking."
        ),
        "seed": RNG_SEED,
        "views": views,
        "summary": {
            key: _summary(value, error_ids[key]) for key, value in errors.items()
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for key, stats in result["summary"].items():
        print(
            f"{key:>16}: n={stats['n']:4d} "
            f"bias={stats['bias_norm_px']:.3f} px "
            f"RMS={stats['rms_px']:.3f} px "
            f"centered={stats['centered_rms_px']:.3f} px "
            f"P95={stats['p95_px']:.3f} px"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
